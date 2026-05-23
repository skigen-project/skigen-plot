import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

const features = [
    {
        title: 'GPU-Accelerated',
        description:
            'Vulkan, Metal, and Direct3D 12 rendering via Qt RHI. 1M+ data points at 60 Hz — no software fallback.',
    },
    {
        title: 'Eigen-Native',
        description:
            'Pass any Eigen::MatrixBase expression directly. No manual data conversion, no copies for contiguous vectors.',
    },
    {
        title: 'Real-Time Ready',
        description:
            'Dynamic vertex buffers with automatic growth. Designed for EEG telemetry, sensor dashboards, and live simulations.',
    },
];

const codeExample = `#include <skigen/plot/plotview.h>
#include <Eigen/Core>
#include <QApplication>
#include <numbers>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(500, 0.f,
        4.f * std::numbers::pi_v<float>);
    Eigen::VectorXf y = x.array().sin();

    Skigen::Plot::PlotView view;
    view.plot(x, y);
    view.show();

    return app.exec();
}`;

function HexGrid() {
    return (
        <div className={styles.hexGrid} aria-hidden="true">
            <svg className={styles.hexSvg} viewBox="0 0 1440 500" preserveAspectRatio="none">
                <defs>
                    <linearGradient id="gridGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.08" />
                        <stop offset="100%" stopColor="#a855f7" stopOpacity="0.08" />
                    </linearGradient>
                </defs>
                <pattern id="hexPattern" x="0" y="0" width="60" height="52" patternUnits="userSpaceOnUse" patternTransform="rotate(0)">
                    <path d="M30,0 L60,15 L60,37 L30,52 L0,37 L0,15 Z" fill="none" stroke="url(#gridGrad)" strokeWidth="0.5" />
                </pattern>
                <rect width="100%" height="100%" fill="url(#hexPattern)" className={styles.hexPatternRect} />
                <circle cx="200" cy="150" r="120" fill="url(#gridGrad)" opacity="0.3" className={styles.orb1} />
                <circle cx="1200" cy="350" r="160" fill="url(#gridGrad)" opacity="0.2" className={styles.orb2} />
            </svg>
        </div>
    );
}

export default function Home(): React.JSX.Element {
    const { siteConfig } = useDocusaurusContext();
    return (
        <Layout title="Home" description={siteConfig.tagline}>
            <header className={styles.hero}>
                <HexGrid />
                <div className={styles.heroInner}>
                    <Heading as="h1" className={styles.heroTitle}>SkigenPlot</Heading>
                    <p className={styles.heroTagline}>{siteConfig.tagline}</p>
                    <div className={styles.heroCta}>
                        <Link className="button button--primary button--lg" to="/docs/guide/overview">
                            Get Started
                        </Link>
                        <Link className={clsx('button button--lg', styles.btnOutline)} to="https://github.com/skigen-project/skigen-plot">
                            View on GitHub
                        </Link>
                    </div>
                </div>
            </header>

            <main>
                {/* Features */}
                <section className={styles.features}>
                    <div className="container">
                        <div className="row">
                            {features.map((f, idx) => (
                                <div key={idx} className="col col--4">
                                    <div className={styles.featureCard}>
                                        <Heading as="h3" className={styles.featureTitle}>{f.title}</Heading>
                                        <p className={styles.featureDesc}>{f.description}</p>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* Code example */}
                <section className={styles.codeSection}>
                    <div className="container">
                        <div className="row">
                            <div className="col col--5">
                                <Heading as="h2" className={styles.codeSectionTitle}>
                                    matplotlib for C++
                                </Heading>
                                <p className={styles.codeSectionDesc}>
                                    A single <code>#include</code>, an Eigen vector, and a call to{' '}
                                    <code>plot()</code> — that's it. Hardware-accelerated rendering from
                                    native C++ types, no Python round-trip.
                                </p>
                                <Link className="button button--primary" to="/docs/guide/line-plot">
                                    Explore Plotting →
                                </Link>
                            </div>
                            <div className="col col--7">
                                <div className={styles.codeBlock}>
                                    <div className={styles.codeHeader}>
                                        <span className={styles.codeDot} style={{ background: '#ff5f57' }} />
                                        <span className={styles.codeDot} style={{ background: '#febc2e' }} />
                                        <span className={styles.codeDot} style={{ background: '#28c840' }} />
                                        <span className={styles.codeFilename}>sine_plot.cpp</span>
                                    </div>
                                    <pre className={styles.codePre}><code>{codeExample}</code></pre>
                                </div>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Stats */}
                <section className={styles.stats}>
                    <div className="container">
                        <div className={styles.statsGrid}>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>60 Hz+</span>
                                <span className={styles.statLabel}>at 1M points</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>C++23</span>
                                <span className={styles.statLabel}>Modern standard</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>GPU</span>
                                <span className={styles.statLabel}>Vulkan / Metal / D3D12</span>
                            </div>
                            <div className={styles.statItem}>
                                <span className={styles.statNumber}>Eigen</span>
                                <span className={styles.statLabel}>Native integration</span>
                            </div>
                        </div>
                    </div>
                </section>
            </main>
        </Layout>
    );
}
