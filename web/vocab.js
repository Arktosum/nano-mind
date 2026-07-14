/* vocab.js — the 75-character vocabulary and tokenizer maps.
   Extracted verbatim from the Python training script. */

const VOCAB =
  "\n !()*,-.03:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyzù—‘’“”";

const STOI = {};
for (let i = 0; i < VOCAB.length; i++) STOI[VOCAB[i]] = i;

/** char -> id (unknown chars are skipped by the caller) */
function encode(text) {
  const ids = [];
  for (const ch of text) if (ch in STOI) ids.push(STOI[ch]);
  return ids;
}

/** id -> char */
function decode(id) {
  return VOCAB[id] ?? "";
}

/** pretty label for a char (spaces / newlines are otherwise invisible) */
function glyph(ch) {
  if (ch === " ") return "␣";
  if (ch === "\n") return "⏎";
  return ch;
}
