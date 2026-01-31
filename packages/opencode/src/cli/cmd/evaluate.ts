import type { Argv } from "yargs"
import { cmd } from "./cmd"
import { spawn } from "child_process"
import path from "path"

interface MoleculeInput {
  name?: string
  mw: number
  logP: number
  tpsa: number
  hbd: number
  hba: number
  rotBonds?: number
  pgpSubstrate?: boolean
}

interface EvaluationResult {
  molecule: string
  risk: number
  confidence: number
  prediction: "APPROVED" | "FAILED"
  reasoning: string[]
  laws_violated: string[]
}

export const EvaluateCommand = cmd({
  command: "evaluate",
  describe: "evaluate pharmaceutical molecules for failure risk using QENEX LAB physics engine",
  builder: (yargs: Argv) => {
    return yargs
      .option("molecule", {
        alias: "m",
        describe: "Molecule properties as JSON or comma-separated (MW,logP,TPSA,HBD,HBA)",
        type: "string",
      })
      .option("name", {
        alias: "n",
        describe: "Molecule name",
        type: "string",
        default: "Unknown",
      })
      .option("file", {
        alias: "f",
        describe: "JSON file containing molecules to evaluate",
        type: "string",
      })
      .option("threshold", {
        alias: "t",
        describe: "Risk threshold for failure prediction (default: 0.35)",
        type: "number",
        default: 0.35,
      })
      .option("demo", {
        alias: "d",
        describe: "Run demonstration with known drugs",
        type: "boolean",
        default: false,
      })
      .option("json", {
        describe: "Output results as JSON",
        type: "boolean",
        default: false,
      })
      .example("$0 evaluate --molecule '441.5,4.2,84.9,2,6' --name Troglitazone", "Evaluate a single molecule")
      .example("$0 evaluate --demo", "Run demo with known approved/failed drugs")
      .example("$0 evaluate --file molecules.json", "Evaluate molecules from file")
  },
  handler: async (args) => {
    if (args.demo) {
      await runDemoEvaluation()
      return
    }

    if (args.molecule) {
      const mol = parseMoleculeInput(args.molecule as string, args.name as string)
      if (!mol) {
        console.error("Invalid molecule format. Use: MW,logP,TPSA,HBD,HBA or JSON")
        process.exit(1)
      }
      await evaluateSingleMolecule(mol, args.threshold as number, args.json as boolean)
      return
    }

    if (args.file) {
      await evaluateFromFile(args.file as string, args.threshold as number, args.json as boolean)
      return
    }

    // Interactive mode - show help
    console.log(`
QENEX LAB Molecule Evaluation System
=====================================

Usage:
  qenex evaluate --molecule "MW,logP,TPSA,HBD,HBA" --name "DrugName"
  qenex evaluate --demo
  qenex evaluate --file molecules.json

Examples:
  # Evaluate Troglitazone (known hepatotoxic TZD)
  qenex evaluate -m "441.5,4.2,84.9,2,6" -n "Troglitazone"

  # Evaluate with lower threshold (more sensitive)
  qenex evaluate -m "441.5,4.2,84.9,2,6" -t 0.30

  # Run demonstration
  qenex evaluate --demo

Properties:
  MW   - Molecular Weight (Da)
  logP - Partition Coefficient
  TPSA - Topological Polar Surface Area (Å²)
  HBD  - Hydrogen Bond Donors
  HBA  - Hydrogen Bond Acceptors
`)
  },
})

function parseMoleculeInput(input: string, name: string): MoleculeInput | null {
  // Try JSON first
  if (input.startsWith("{")) {
    try {
      const parsed = JSON.parse(input)
      return {
        name: parsed.name || name,
        mw: parsed.mw || parsed.MW || parsed.molecular_weight,
        logP: parsed.logP || parsed.logp,
        tpsa: parsed.tpsa || parsed.TPSA,
        hbd: parsed.hbd || parsed.HBD || parsed.num_hbd,
        hba: parsed.hba || parsed.HBA || parsed.num_hba,
        rotBonds: parsed.rotBonds || parsed.num_rotatable_bonds,
        pgpSubstrate: parsed.pgpSubstrate || parsed.pgp_substrate,
      }
    } catch {
      return null
    }
  }

  // Try comma-separated
  const parts = input.split(",").map((p) => p.trim())
  if (parts.length >= 5) {
    return {
      name,
      mw: parseFloat(parts[0]),
      logP: parseFloat(parts[1]),
      tpsa: parseFloat(parts[2]),
      hbd: parseInt(parts[3]),
      hba: parseInt(parts[4]),
      rotBonds: parts[5] ? parseInt(parts[5]) : undefined,
    }
  }

  return null
}

async function evaluateSingleMolecule(mol: MoleculeInput, threshold: number, jsonOutput: boolean): Promise<void> {
  const result = await callPythonEvaluator([mol], threshold)

  if (jsonOutput) {
    console.log(JSON.stringify(result, null, 2))
    return
  }

  displayResult(result[0])
}

async function evaluateFromFile(filePath: string, threshold: number, jsonOutput: boolean): Promise<void> {
  const fs = await import("fs")
  const content = fs.readFileSync(filePath, "utf-8")
  const molecules: MoleculeInput[] = JSON.parse(content)

  const results = await callPythonEvaluator(molecules, threshold)

  if (jsonOutput) {
    console.log(JSON.stringify(results, null, 2))
    return
  }

  console.log("\n" + "=".repeat(70))
  console.log("QENEX LAB BATCH EVALUATION RESULTS")
  console.log("=".repeat(70) + "\n")

  for (const result of results) {
    displayResult(result)
    console.log("-".repeat(70))
  }

  // Summary
  const failed = results.filter((r) => r.prediction === "FAILED").length
  const approved = results.filter((r) => r.prediction === "APPROVED").length

  console.log(`\nSUMMARY: ${approved} APPROVED, ${failed} FAILED out of ${results.length} molecules`)
}

async function runDemoEvaluation(): Promise<void> {
  const workspaceRoot = path.resolve(__dirname, "../../../../../workspace")
  const scriptPath = path.join(workspaceRoot, "packages/qenex-tissue/src/blind_validation.py")

  return new Promise((resolve, reject) => {
    const proc = spawn("python", ["-m", "src.blind_validation"], {
      cwd: path.join(workspaceRoot, "packages/qenex-tissue"),
      stdio: "inherit",
    })

    proc.on("close", (code) => {
      if (code === 0) {
        resolve()
      } else {
        reject(new Error(`Demo exited with code ${code}`))
      }
    })

    proc.on("error", (err) => {
      reject(err)
    })
  })
}

async function callPythonEvaluator(molecules: MoleculeInput[], threshold: number): Promise<EvaluationResult[]> {
  const workspaceRoot = path.resolve(__dirname, "../../../../../workspace")

  const pythonScript = `
import sys
import json

sys.path.insert(0, '${workspaceRoot}/packages/qenex-tissue')
from src.blind_validation import BlindValidationProtocol

protocol = BlindValidationProtocol()
molecules = json.loads('''${JSON.stringify(molecules)}''')

for i, mol in enumerate(molecules):
    protocol.add_molecule(
        molecule_id=f"MOL_{i}",
        name=mol.get("name", f"Molecule_{i}"),
        molecular_weight=mol["mw"],
        logP=mol["logP"],
        tpsa=mol["tpsa"],
        num_hbd=mol["hbd"],
        num_hba=mol["hba"],
        num_rotatable_bonds=mol.get("rotBonds", 0),
        pgp_substrate=mol.get("pgpSubstrate", False)
    )

protocol.predict_all(threshold=${threshold})

results = []
for mol in protocol.molecules:
    results.append({
        "molecule": mol.name,
        "risk": mol.predicted_failure_risk,
        "confidence": mol.prediction_confidence,
        "prediction": mol.predicted_outcome.upper() if mol.predicted_outcome else "UNKNOWN",
        "reasoning": mol.prediction_reasoning.split("; ") if mol.prediction_reasoning else []
    })

print(json.dumps(results))
`

  return new Promise((resolve, reject) => {
    const proc = spawn("python", ["-c", pythonScript], {
      cwd: workspaceRoot,
    })

    let stdout = ""
    let stderr = ""

    proc.stdout.on("data", (data) => {
      stdout += data.toString()
    })

    proc.stderr.on("data", (data) => {
      stderr += data.toString()
    })

    proc.on("close", (code) => {
      if (code === 0) {
        try {
          const results = JSON.parse(stdout.trim())
          resolve(results)
        } catch (e) {
          reject(new Error(`Failed to parse results: ${stdout}\n${stderr}`))
        }
      } else {
        reject(new Error(`Python evaluator failed: ${stderr}`))
      }
    })

    proc.on("error", (err) => {
      reject(err)
    })
  })
}

function displayResult(result: EvaluationResult): void {
  const riskBar = createRiskBar(result.risk)
  const predColor = result.prediction === "FAILED" ? "\x1b[31m" : "\x1b[32m"
  const reset = "\x1b[0m"

  console.log(`
┌─────────────────────────────────────────────────────────────────────┐
│ MOLECULE: ${result.molecule.padEnd(56)} │
├─────────────────────────────────────────────────────────────────────┤
│ Prediction: ${predColor}${result.prediction.padEnd(10)}${reset}                                          │
│ Risk Score: ${(result.risk * 100).toFixed(1).padStart(5)}%  ${riskBar}                     │
│ Confidence: ${(result.confidence * 100).toFixed(1).padStart(5)}%                                              │
├─────────────────────────────────────────────────────────────────────┤
│ Risk Factors:                                                       │`)

  if (result.reasoning.length === 0 || (result.reasoning.length === 1 && result.reasoning[0].includes("No significant"))) {
    console.log("│   No significant risk factors identified                          │")
  } else {
    for (const reason of result.reasoning.slice(0, 5)) {
      const truncated = reason.length > 63 ? reason.substring(0, 60) + "..." : reason
      console.log(`│   - ${truncated.padEnd(62)} │`)
    }
    if (result.reasoning.length > 5) {
      console.log(`│   ... and ${result.reasoning.length - 5} more factors                                      │`)
    }
  }

  console.log("└─────────────────────────────────────────────────────────────────────┘")
}

function createRiskBar(risk: number): string {
  const filled = Math.round(risk * 20)
  const empty = 20 - filled

  let color = "\x1b[32m" // green
  if (risk > 0.6) color = "\x1b[31m" // red
  else if (risk > 0.35) color = "\x1b[33m" // yellow

  const reset = "\x1b[0m"
  return `[${color}${"█".repeat(filled)}${reset}${"░".repeat(empty)}]`
}
