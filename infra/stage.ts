export const domain = (() => {
  if ($app.stage === "production") return "qenex.ai"
  if ($app.stage === "dev") return "dev.qenex.ai"
  return `${$app.stage}.dev.qenex.ai`
})()

export const zoneID = "430ba34c138cfb5360826c4909f99be8"

new cloudflare.RegionalHostname("RegionalHostname", {
  hostname: domain,
  regionKey: "us",
  zoneId: zoneID,
})

export const shortDomain = (() => {
  if ($app.stage === "production") return "qnx.ai"
  if ($app.stage === "dev") return "dev.qnx.ai"
  return `${$app.stage}.dev.qnx.ai`
})()
