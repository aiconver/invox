const typescriptTransform = require('i18next-scanner-typescript');
const chalk = require('chalk');
const fs = require('fs');

module.exports = {
    input: [
        'src/**/*.{js,jsx,ts,tsx}',
        // Use ! to filter out files or directories
        '!src/**/*.spec.{js,jsx}',
        '!src/i18n/**',
        '!**/node_modules/**',
    ],
    output: './',
    options: {
        debug: true,
        func: {
            list: ['i18next.t', 'i18n.t'],
            extensions: ['.js', '.jsx']
        },
        trans: {
            component: 'Trans',
            i18nKey: 'i18nKey',
            defaultsKey: 'defaults',
            extensions: ['.js', '.jsx'],
            fallbackKey: function(ns, value) {
                return value;
            },

            // https://react.i18next.com/latest/trans-component#usage-with-simple-html-elements-like-less-than-br-greater-than-and-others-v10.4.0
            supportBasicHtmlNodes: true, // Enables keeping the name of simple nodes (e.g. <br/>) in translations instead of indexed keys.
            keepBasicHtmlNodesFor: ['br', 'strong', 'i', 'p'], // Which nodes are allowed to be kept in translations during defaultValue generation of <Trans>.

            // https://github.com/acornjs/acorn/tree/master/acorn#interface
            acorn: {
                ecmaVersion: 2020,
                sourceType: 'module', // defaults to 'module'
            }
        },
        lngs: ['en','de'],
        ns: [
            'locale',
            'resource',
            'translations'
        ],
        defaultLng: 'en',
        defaultNs: 'resource',
        defaultValue: '__STRING_NOT_TRANSLATED__',
        resource: {
            loadPath: 'i18n/{{lng}}/{{ns}}.json',
            savePath: 'i18n/{{lng}}/{{ns}}.json',
            jsonIndent: 2,
            lineEnding: '\n'
        },
        nsSeparator: true, // namespace separator
        interpolation: {
            prefix: '{{',
            suffix: '}}'
        },
        metadata: {},
        allowDynamicKeys: false,
    },
    transform: typescriptTransform(
		  // options
		  {
		    // default value for extensions
		    extensions: [".ts", ".tsx"],
		    // optional ts configuration
		    tsOptions: {
		      target: "es2017",
		    },
		  }
    ),    
};
