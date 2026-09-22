# Public v2 migration scope

This branch upgrades the existing public `Open-LLM-VTuber-Rinne` project from
the validated public release `0c66f0f3`. It does not publish the private
repository or its history.

## Source checkpoints

- Public base: `0c66f0f3` (`desktop-v1.2.1-rinne.1`)
- Private implementation reference: `8735e324`
- Current Electron/Web implementation reference: `3c32805`

The private checkpoints are references for selective migration only. They must
not be merged wholesale into this branch.

This local migration branch is transitional. The packaged Rinne model and voice
references inherited from the previous public release remain temporarily so
that the first checkpoint still starts normally. They are not approved for the
final public v2 tree and will be removed only when the external asset setup path
replaces their runtime references in the same runnable commit.

## Included in the public release

- Application source code and generic memory functionality
- Generic local-library code, with no library contents
- Sanitized configuration templates
- Minimal-neutral Rinne prompts
- Author-created outfits under the separate non-commercial asset license
- External game-asset loading interfaces and local setup documentation
- Reproducible build and installation instructions

## Excluded from the public release

- API keys, account identifiers, machine-specific paths, private configuration
- Personal chats, diaries, memories, QQ data and local library contents
- Private tests, answer keys, research reports and handoff documents
- Personal forms of address and relationship-specific instructions
- Sexualized punishment or other private-preference instructions
- Original or converted game assets and game-derived acceptance composites
- Locally built `app.asar`, installers, caches and runtime logs

## Prompt migration rule

Prompt changes must be minimal. Replace only strongly personal wording, such as
private names, forms of address, memories and preferences. Use `用户` when a
neutral subject is required, then adjust only the nearby words needed to keep
the sentence natural. Preserve paragraph order, rule strength, output schema
and all unrelated wording.

The public personality should remain warm, sincere, gentle and slightly shy,
while respecting the user's boundaries. Private punishment, dominance or
sexual-preference instructions are replaced with ordinary comforting,
encouraging and affectionate interaction.

## Migration phases

1. Establish public data boundaries and directory contracts.
2. Selectively port backend/runtime fixes from the private checkpoint.
3. Port and minimally sanitize prompts and default configuration.
4. Publish the generic library implementation without user data.
5. Add the local game-resource setup interface and manual fallback.
6. Add author-created outfits and their non-commercial license.
7. Publish matching frontend source and compiled frontend artifacts.
8. Build from a fresh clone and verify Live Mode, Pet Mode, outfits and local
   game-Rinne loading.

No GitHub branch, release or artifact is updated until the local public build
passes review and the owner explicitly approves publication.
