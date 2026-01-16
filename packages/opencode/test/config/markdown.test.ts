import { expect, test } from "bun:test"
import { ConfigMarkdown } from "../../src/config/markdown"

const template = `This is a @valid/path/to/a/file and it should also match at
the beginning of a line:

@another-valid/path/to/a/file

but this is not:

   - Adds a "Co-authored-by:" footer which clarifies which AI agent
     helped create this commit, using an appropriate \`noreply@...\`
     or \`noreply@anthropic.com\` email address.

We also need to deal with files followed by @commas, ones
with @file-extensions.md, even @multiple.extensions.bak,
hidden directories like @.config/ or files like @.bashrc
and ones at the end of a sentence like @foo.md.

Also shouldn't forget @/absolute/paths.txt with and @/without/extensions,
as well as @~/home-files and @~/paths/under/home.txt.

If the reference is \`@quoted/in/backticks\` then it shouldn't match at all.`

const matches = ConfigMarkdown.files(template)

test("should extract exactly 12 file references", () => {
  expect(matches.length).toBe(12)
})

test("should extract valid/path/to/a/file", () => {
  expect(matches[0][1]).toBe("valid/path/to/a/file")
})

test("should extract another-valid/path/to/a/file", () => {
  expect(matches[1][1]).toBe("another-valid/path/to/a/file")
})

test("should extract paths ignoring comma after", () => {
  expect(matches[2][1]).toBe("commas")
})

test("should extract a path with a file extension and comma after", () => {
  expect(matches[3][1]).toBe("file-extensions.md")
})

test("should extract a path with multiple dots and comma after", () => {
  expect(matches[4][1]).toBe("multiple.extensions.bak")
})

test("should extract hidden directory", () => {
  expect(matches[5][1]).toBe(".config/")
})

test("should extract hidden file", () => {
  expect(matches[6][1]).toBe(".bashrc")
})

test("should extract a file ignoring period at end of sentence", () => {
  expect(matches[7][1]).toBe("foo.md")
})

test("should extract an absolute path with an extension", () => {
  expect(matches[8][1]).toBe("/absolute/paths.txt")
})

test("should extract an absolute path without an extension", () => {
  expect(matches[9][1]).toBe("/without/extensions")
})

test("should extract an absolute path in home directory", () => {
  expect(matches[10][1]).toBe("~/home-files")
})

test("should extract an absolute path under home directory", () => {
  expect(matches[11][1]).toBe("~/paths/under/home.txt")
})

test("should not match when preceded by backtick", () => {
  const backtickTest = "This `@should/not/match` should be ignored"
  const backtickMatches = ConfigMarkdown.files(backtickTest)
  expect(backtickMatches.length).toBe(0)
})

test("should not match email addresses", () => {
  const emailTest = "Contact user@example.com for help"
  const emailMatches = ConfigMarkdown.files(emailTest)
  expect(emailMatches.length).toBe(0)
})

// Tests for shell() function
test("should extract shell commands from template", () => {
  const template = "Run this command: !`echo hello` and also !`git status`"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(2)
  expect(matches[0][1]).toBe("echo hello")
  expect(matches[1][1]).toBe("git status")
})

test("should return empty array when no shell commands", () => {
  const template = "No shell commands here, just @some/file/path"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(0)
})

test("should extract single shell command", () => {
  const template = "Current branch: !`git branch --show-current`"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(1)
  expect(matches[0][1]).toBe("git branch --show-current")
})

test("should extract commands with pipes and special characters", () => {
  const template = "Files: !`ls -la | grep .ts | head -5`"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(1)
  expect(matches[0][1]).toBe("ls -la | grep .ts | head -5")
})

test("should handle multiple shell commands on same line", () => {
  const template = "Date: !`date`, User: !`whoami`"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(2)
  expect(matches[0][1]).toBe("date")
  expect(matches[1][1]).toBe("whoami")
})

test("should not match empty backticks", () => {
  const template = "This is !`` an empty command"
  const matches = ConfigMarkdown.shell(template)
  expect(matches.length).toBe(0)
})

test("should handle multiline template with shell commands", () => {
  const template = `First line
Run: !${"git status"}
More text
Also: !${"npm test"}`
  // Note: We need to use template backticks in a way that doesn't confuse the regex
  const template2 = "First line\nRun: !`git status`\nMore text\nAlso: !`npm test`"
  const matches = ConfigMarkdown.shell(template2)
  expect(matches.length).toBe(2)
  expect(matches[0][1]).toBe("git status")
  expect(matches[1][1]).toBe("npm test")
})
