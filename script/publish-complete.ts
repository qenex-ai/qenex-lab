#!/usr/bin/env bun

import { Script } from "@qenex-lab/script"
import { $ } from "bun"

if (!Script.preview) {
  await $`gh release edit v${Script.version} --draft=false`
}

await $`bun install`

await $`gh release download --pattern "qenex-linux-*64.tar.gz" --pattern "qenex-darwin-*64.zip" -D dist`

await import(`../packages/opencode/script/publish-registries.ts`)
