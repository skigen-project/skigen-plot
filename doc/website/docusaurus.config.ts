import { themes as prismThemes } from 'prism-react-renderer';
import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
    title: 'SkigenPlot',
    tagline: 'Hardware-accelerated C++ plotting for the scientific and ML ecosystem.',
    favicon: 'img/skigen-logo_plain.svg',

    url: 'https://skigen-project.github.io',
    baseUrl: '/skigen-plot/',

    organizationName: 'skigen-project',
    projectName: 'skigen-plot-doc',

    onBrokenLinks: 'warn',

    i18n: {
        defaultLocale: 'en',
        locales: ['en'],
    },

    presets: [
        [
            'classic',
            {
                docs: {
                    sidebarPath: './sidebars.ts',
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                },
                blog: false,
                theme: {
                    customCss: './src/css/custom.css',
                },
            } satisfies Preset.Options,
        ],
    ],

    themes: [
        '@docusaurus/theme-mermaid',
        [
            '@easyops-cn/docusaurus-search-local',
            {
                hashed: true,
                indexDocs: true,
                indexBlog: false,
                docsRouteBasePath: '/docs',
                searchBarShortcutHint: true,
                searchBarPosition: 'right',
            },
        ],
    ],

    markdown: {
        format: 'detect',
        mermaid: true,
        hooks: {
            onBrokenMarkdownLinks: 'warn',
        },
    },

    stylesheets: [
        '/skigen-plot/katex.min.css',
    ],

    themeConfig: {
        colorMode: {
            defaultMode: 'dark',
            disableSwitch: false,
            respectPrefersColorScheme: true,
        },
        navbar: {
            title: 'SkigenPlot',
            logo: {
                alt: 'SkigenPlot Logo',
                src: 'img/skigen-logo.svg',
                srcDark: 'img/skigen-logo-dark.svg',
            },
            items: [
                {
                    type: 'docSidebar',
                    sidebarId: 'guideSidebar',
                    position: 'left',
                    label: 'Guide',
                },
                {
                    type: 'docSidebar',
                    sidebarId: 'apiSidebar',
                    position: 'left',
                    label: 'API',
                },
                {
                    href: 'https://skigen-project.github.io/',
                    label: 'Skigen',
                    position: 'right',
                },
                {
                    href: 'https://github.com/skigen-project/skigen-plot',
                    position: 'right',
                    className: 'header-github-link',
                    'aria-label': 'GitHub repository',
                },
            ],
        },
        footer: {
            style: 'dark',
            links: [
                {
                    title: 'Guide',
                    items: [
                        { label: 'Overview', to: '/docs/guide/overview' },
                        { label: 'Getting Started', to: '/docs/guide/getting-started' },
                    ],
                },
                {
                    title: 'API',
                    items: [
                        { label: 'PlotView', to: '/docs/api/plotview' },
                        { label: 'Camera3D', to: '/docs/api/camera3d' },
                    ],
                },
                {
                    title: 'Skigen Ecosystem',
                    items: [
                        { label: 'Skigen', href: 'https://skigen-project.github.io/' },
                        { label: 'GitHub', href: 'https://github.com/skigen-project/skigen-plot' },
                    ],
                },
            ],
            copyright: `Copyright © ${new Date().getFullYear()} Skigen Contributors.`,
        },
        prism: {
            theme: prismThemes.github,
            darkTheme: prismThemes.dracula,
            additionalLanguages: ['cpp', 'cmake', 'bash'],
        },
    } satisfies Preset.ThemeConfig,
};

export default config;
