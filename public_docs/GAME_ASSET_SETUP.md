# Local game-resource setup contract

The public repository and its release packages do not contain Rinne game
resources or converted runtime bundles.

The completed setup flow will:

1. Ask the user to select the relevant source files from their own computer.
2. Validate that the selected files match a supported layout and version.
3. Perform all conversion locally without uploading source files or paths.
4. Write generated runtime data outside the Git working tree.
5. Store only local configuration that points the application at that data.
6. Refuse unsupported or incomplete inputs with a readable error.
7. Provide an equivalent manual configuration procedure.

The clean-install acceptance target is that a user can clone the public
project, follow the documented local procedure, and load the supported game
Rinne presentation in both Live Mode and Pet Mode. The setup must also support
removing and rebuilding the generated local data without modifying source
files.

This file defines the interface boundary. The converter and UI will be added in
a later migration phase.
