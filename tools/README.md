# tools/

Host-side developer tools: GGUF inspectors, graph dumpers, and logit diffing
against `llama.cpp`. Tools may use heavier dependencies than the header-only
library, but nothing here is a dependency of `branes::synapse`.

Each tool will get a `CMakeLists.txt` entry when the first one lands; the
directory is not yet part of the build.
