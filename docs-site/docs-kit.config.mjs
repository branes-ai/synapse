// The docs site's configuration: everything docs-kit does not default.
// https://github.com/branes-ai/docs-kit
export default {
  repo: 'synapse',
  title: 'Synapse',
  description: 'The ML runtime of the Branes.AI platform: a header-only C++20 tensor graph engine that loads GGUF models, plans their memory statically, and runs them on MTL5 kernels.',
  // Kept in the repo, not published: a link to one goes to GitHub.
  unpublished: ['sessions'],
  accent: {
    dark: { low: '#2a1f4d', accent: '#9b7bff', high: '#ddd3ff' },
    light: { low: '#e6defe', accent: '#6440d8', high: '#2a1a63' },
  },
  api: {
    name: 'Synapse C++ API',
    brief: 'branes::synapse — GGUF loader, quantized kernels, graph engine and models',
    input: ['sdk/include'],
  },
};
