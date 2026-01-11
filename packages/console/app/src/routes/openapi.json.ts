export async function GET() {
  const response = await fetch(
    "https://raw.githubusercontent.com/abdulrahman305/qenex-lab/refs/heads/master/packages/sdk/openapi.json",
  )
  const json = await response.json()
  return json
}
