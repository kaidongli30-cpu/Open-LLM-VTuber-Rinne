# Rinne game-asset setup interface

This directory is reserved for the local-only setup interface that connects a
user-selected game source to the external Rinne runtime loader.

The implementation must not embed, download or upload game resources. It must
validate inputs, write generated data outside the repository, produce a local
configuration file, support a manual fallback, and fail without modifying the
source files when validation does not pass.

The concrete converter and setup UI will be introduced only after their input,
output and rollback contracts are covered by the public migration plan.
