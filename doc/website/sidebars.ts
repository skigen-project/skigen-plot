import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
    guideSidebar: [
        'guide/overview',
        'guide/getting-started',
        {
            type: 'category',
            label: '2D Plotting',
            items: [
                'guide/line-plot',
                'guide/scatter-plot',
            ],
        },
        {
            type: 'category',
            label: '3D Rendering',
            items: [
                'guide/camera3d',
            ],
        },
        'guide/realtime',
    ],
    apiSidebar: [
        {
            type: 'category',
            label: 'Skigen::Plot',
            items: [
                'api/plotview',
                'api/camera3d',
                'api/boundingbox',
            ],
        },
    ],
};

export default sidebars;
