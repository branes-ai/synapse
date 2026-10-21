// commitlint configuration — Conventional Commits, enforced in CI.
// See: https://www.conventionalcommits.org/en/v1.0.0/

module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        // Allowed CC types. Order = preference for changelog grouping.
        'type-enum': [
            2,
            'always',
            [
                'feat',     // new user-visible feature
                'fix',      // bug fix
                'build',    // build system / dependencies
                'ci',       // CI/CD configuration
                'chore',    // tooling / housekeeping
                'docs',     // documentation only
                'test',     // tests only
                'refactor', // code change that is neither a feat nor a fix
                'perf',     // performance improvement
                'style',    // formatting / whitespace only
                'revert',   // revert of a previous commit
            ],
        ],
        // PR titles can be a bit longer than the typical 72.
        'header-max-length': [2, 'always', 100],
        // Allow lowercase scopes only — keeps changelog clean.
        'scope-case': [2, 'always', 'lower-case'],
    },
};
