"""
Feature Quality Analysis
特徴量の品質を詳細に分析し、問題点を特定する
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import json

# プロジェクトルートをパスに追加
BASE_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(BASE_DIR))


def load_data():
    """データを読み込み"""
    DATA_DIR = BASE_DIR / "data" / "processed"

    with open(DATA_DIR / "linear_features_train.pkl", 'rb') as f:
        train_data = pickle.load(f)

    with open(DATA_DIR / "feature_groups.pkl", 'rb') as f:
        feature_groups = pickle.load(f)

    with open(DATA_DIR / "dimension_metadata.json", 'r') as f:
        dimension_metadata = json.load(f)

    return train_data, feature_groups, dimension_metadata


def analyze_feature_statistics(X, group_name, group_info):
    """特徴グループの統計を分析"""
    start = group_info['start']
    end = group_info['end']
    X_group = X[:, start:end]

    stats = {
        'group_name': group_name,
        'size': group_info['size'],
        'mean': float(np.mean(X_group)),
        'std': float(np.std(X_group)),
        'min': float(np.min(X_group)),
        'max': float(np.max(X_group)),
        'zeros': float(np.sum(X_group == 0) / X_group.size * 100),
        'near_zeros': float(np.sum(np.abs(X_group) < 1e-6) / X_group.size * 100),
        'range': float(np.max(X_group) - np.min(X_group)),
        'median': float(np.median(X_group)),
        'p25': float(np.percentile(X_group, 25)),
        'p75': float(np.percentile(X_group, 75)),
    }

    # Sparsity
    stats['sparsity'] = stats['near_zeros']

    # Scale (典型的な値の大きさ)
    stats['typical_scale'] = float(np.percentile(np.abs(X_group.flatten()), 75))

    return stats


def analyze_feature_correlations(X, y, group_name, group_info, top_k=5):
    """特徴とターゲットの相関を分析"""
    start = group_info['start']
    end = group_info['end']
    X_group = X[:, start:end]

    correlations = []
    for i in range(X_group.shape[1]):
        feat = X_group[:, i]

        # Skip if constant
        if np.std(feat) < 1e-10:
            corr = 0.0
        else:
            # Spearman correlation
            try:
                corr, _ = spearmanr(feat, y)
                if np.isnan(corr):
                    corr = 0.0
            except:
                corr = 0.0

        correlations.append({
            'dim': start + i,
            'correlation': float(abs(corr)),
            'correlation_signed': float(corr)
        })

    # Sort by absolute correlation
    correlations.sort(key=lambda x: x['correlation'], reverse=True)

    return {
        'top_correlations': correlations[:top_k],
        'mean_abs_corr': float(np.mean([c['correlation'] for c in correlations])),
        'max_abs_corr': float(correlations[0]['correlation']) if correlations else 0.0,
        'n_significant': sum(1 for c in correlations if c['correlation'] > 0.1)
    }


def compare_feature_groups(stats_list):
    """特徴グループ間の比較"""
    print("\n" + "=" * 80)
    print("FEATURE GROUPS COMPARISON")
    print("=" * 80)

    # Sort by typical scale
    stats_list_sorted = sorted(stats_list, key=lambda x: x['stats']['typical_scale'], reverse=True)

    print("\nBy Typical Scale (影響力の大きさ):")
    print(f"{'Group':<30s} {'Scale':>10s} {'Mean':>10s} {'Std':>10s} {'Sparsity':>10s}")
    print("-" * 80)

    for item in stats_list_sorted:
        s = item['stats']
        print(f"{s['group_name']:<30s} "
              f"{s['typical_scale']:>10.4f} "
              f"{s['mean']:>10.4f} "
              f"{s['std']:>10.4f} "
              f"{s['sparsity']:>9.1f}%")

    print("\n" + "=" * 80)
    print("SCALE RATIO ANALYSIS")
    print("=" * 80)

    # Calculate scale ratios
    scales = [s['stats']['typical_scale'] for s in stats_list_sorted]
    max_scale = max(scales)
    min_scale = min([s for s in scales if s > 0])

    print(f"\nMax scale: {max_scale:.4f}")
    print(f"Min scale: {min_scale:.6f}")
    print(f"Ratio (max/min): {max_scale / min_scale:.1f}x")

    print("\n⚠️  Scale差が大きいと、小さいスケールの特徴が無視される可能性があります")


def analyze_user_vs_movie_features(X, y, feature_groups):
    """ユーザー特徴 vs 映画特徴の比較"""
    print("\n" + "=" * 80)
    print("USER vs MOVIE FEATURES")
    print("=" * 80)

    user_groups = [g for g in feature_groups['groups'].keys() if g.startswith('user_')]
    movie_groups = [g for g in feature_groups['groups'].keys() if g.startswith('movie_')]
    review_groups = [g for g in feature_groups['groups'].keys() if g.startswith('review_')]

    # Extract features
    user_indices = []
    for g in user_groups:
        info = feature_groups['groups'][g]
        user_indices.extend(range(info['start'], info['end']))

    movie_indices = []
    for g in movie_groups:
        info = feature_groups['groups'][g]
        movie_indices.extend(range(info['start'], info['end']))

    review_indices = []
    for g in review_groups:
        info = feature_groups['groups'][g]
        review_indices.extend(range(info['start'], info['end']))

    X_user = X[:, user_indices]
    X_movie = X[:, movie_indices]
    X_review = X[:, review_indices]

    print("\nUser Features:")
    print(f"  Dimensions: {X_user.shape[1]}")
    print(f"  Mean: {np.mean(X_user):.4f}")
    print(f"  Std: {np.std(X_user):.4f}")
    print(f"  Typical scale: {np.percentile(np.abs(X_user.flatten()), 75):.4f}")
    print(f"  Sparsity: {np.sum(np.abs(X_user) < 1e-6) / X_user.size * 100:.1f}%")

    print("\nMovie Features:")
    print(f"  Dimensions: {X_movie.shape[1]}")
    print(f"  Mean: {np.mean(X_movie):.4f}")
    print(f"  Std: {np.std(X_movie):.4f}")
    print(f"  Typical scale: {np.percentile(np.abs(X_movie.flatten()), 75):.4f}")
    print(f"  Sparsity: {np.sum(np.abs(X_movie) < 1e-6) / X_movie.size * 100:.1f}%")

    print("\nReview Features:")
    print(f"  Dimensions: {X_review.shape[1]}")
    print(f"  Mean: {np.mean(X_review):.4f}")
    print(f"  Std: {np.std(X_review):.4f}")
    print(f"  Typical scale: {np.percentile(np.abs(X_review.flatten()), 75):.4f}")
    print(f"  Sparsity: {np.sum(np.abs(X_review) < 1e-6) / X_review.size * 100:.1f}%")

    # Correlation with target
    print("\nCorrelation with target (y):")

    user_corrs = []
    for i in range(X_user.shape[1]):
        if np.std(X_user[:, i]) > 1e-10:
            try:
                corr, _ = spearmanr(X_user[:, i], y)
                if not np.isnan(corr):
                    user_corrs.append(abs(corr))
            except:
                pass

    movie_corrs = []
    for i in range(X_movie.shape[1]):
        if np.std(X_movie[:, i]) > 1e-10:
            try:
                corr, _ = spearmanr(X_movie[:, i], y)
                if not np.isnan(corr):
                    movie_corrs.append(abs(corr))
            except:
                pass

    review_corrs = []
    for i in range(X_review.shape[1]):
        if np.std(X_review[:, i]) > 1e-10:
            try:
                corr, _ = spearmanr(X_review[:, i], y)
                if not np.isnan(corr):
                    review_corrs.append(abs(corr))
            except:
                pass

    print(f"  User:   Mean |ρ| = {np.mean(user_corrs):.4f}, Max = {np.max(user_corrs) if user_corrs else 0:.4f}")
    print(f"  Movie:  Mean |ρ| = {np.mean(movie_corrs):.4f}, Max = {np.max(movie_corrs) if movie_corrs else 0:.4f}")
    print(f"  Review: Mean |ρ| = {np.mean(review_corrs):.4f}, Max = {np.max(review_corrs) if review_corrs else 0:.4f}")


def detect_problematic_features(X, y, feature_groups, dimension_metadata):
    """問題のある特徴を検出"""
    print("\n" + "=" * 80)
    print("PROBLEMATIC FEATURES DETECTION")
    print("=" * 80)

    issues = []

    for group_name, group_info in feature_groups['groups'].items():
        start = group_info['start']
        end = group_info['end']
        X_group = X[:, start:end]

        # Check 1: Constant features
        for i in range(X_group.shape[1]):
            if np.std(X_group[:, i]) < 1e-10:
                issues.append({
                    'dim': start + i,
                    'group': group_name,
                    'issue': 'constant',
                    'severity': 'high'
                })

        # Check 2: Extreme sparsity (>99%)
        sparsity = np.sum(np.abs(X_group) < 1e-6) / X_group.size * 100
        if sparsity > 99:
            issues.append({
                'dim': f'{start}-{end}',
                'group': group_name,
                'issue': f'extreme_sparsity ({sparsity:.1f}%)',
                'severity': 'medium'
            })

        # Check 3: Extreme scale (too small)
        typical_scale = np.percentile(np.abs(X_group.flatten()), 75)
        if typical_scale < 1e-6:
            issues.append({
                'dim': f'{start}-{end}',
                'group': group_name,
                'issue': f'too_small_scale ({typical_scale:.2e})',
                'severity': 'high'
            })

    # Print issues
    if issues:
        print(f"\n⚠️  Found {len(issues)} potential issues:\n")

        high_severity = [i for i in issues if i['severity'] == 'high']
        medium_severity = [i for i in issues if i['severity'] == 'medium']

        if high_severity:
            print("HIGH SEVERITY:")
            for issue in high_severity[:10]:  # Show first 10
                print(f"  - Dim {issue['dim']} ({issue['group']}): {issue['issue']}")

        if medium_severity:
            print("\nMEDIUM SEVERITY:")
            for issue in medium_severity[:10]:
                print(f"  - Dim {issue['dim']} ({issue['group']}): {issue['issue']}")
    else:
        print("\n✅ No major issues detected")

    return issues


def plot_feature_distributions(X, feature_groups, output_dir):
    """特徴量の分布を可視化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("=" * 80)

    # Plot 1: Scale comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    group_names = []
    scales = []

    for group_name, group_info in feature_groups['groups'].items():
        start = group_info['start']
        end = group_info['end']
        X_group = X[:, start:end]

        typical_scale = np.percentile(np.abs(X_group.flatten()), 75)

        group_names.append(group_name)
        scales.append(typical_scale)

    # Sort by scale
    sorted_indices = np.argsort(scales)[::-1]
    group_names = [group_names[i] for i in sorted_indices]
    scales = [scales[i] for i in sorted_indices]

    ax.barh(range(len(group_names)), scales)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.set_xlabel('Typical Scale (75th percentile of |values|)')
    ax.set_title('Feature Group Scales Comparison')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_scales.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_dir / 'feature_scales.png'}")

    # Plot 2: User vs Movie vs Review
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    user_indices = []
    movie_indices = []
    review_indices = []

    for group_name, group_info in feature_groups['groups'].items():
        start = group_info['start']
        end = group_info['end']

        if group_name.startswith('user_'):
            user_indices.extend(range(start, end))
        elif group_name.startswith('movie_'):
            movie_indices.extend(range(start, end))
        elif group_name.startswith('review_'):
            review_indices.extend(range(start, end))

    X_user = X[:, user_indices].flatten()
    X_movie = X[:, movie_indices].flatten()
    X_review = X[:, review_indices].flatten()

    # Remove zeros for better visualization
    X_user_nonzero = X_user[np.abs(X_user) > 1e-6]
    X_movie_nonzero = X_movie[np.abs(X_movie) > 1e-6]
    X_review_nonzero = X_review[np.abs(X_review) > 1e-6]

    axes[0].hist(X_user_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_title('User Features (non-zero)')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_yscale('log')

    axes[1].hist(X_movie_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title('Movie Features (non-zero)')
    axes[1].set_xlabel('Value')
    axes[1].set_yscale('log')

    axes[2].hist(X_review_nonzero, bins=50, edgecolor='black', alpha=0.7)
    axes[2].set_title('Review Features (non-zero)')
    axes[2].set_xlabel('Value')
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_dir / 'feature_distributions.png'}")


def main():
    print("=" * 80)
    print("FEATURE QUALITY ANALYSIS")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    train_data, feature_groups, dimension_metadata = load_data()

    X = train_data['X']
    y = train_data['y']  # user-normalized

    print(f"  Samples: {X.shape[0]:,}")
    print(f"  Features: {X.shape[1]:,}")

    # Analyze each feature group
    print("\n[2/6] Analyzing feature groups...")
    all_stats = []

    for group_name, group_info in feature_groups['groups'].items():
        stats = analyze_feature_statistics(X, group_name, group_info)
        corr_stats = analyze_feature_correlations(X, y, group_name, group_info)

        all_stats.append({
            'group': group_name,
            'stats': stats,
            'correlations': corr_stats
        })

    # Compare groups
    print("\n[3/6] Comparing feature groups...")
    compare_feature_groups(all_stats)

    # User vs Movie
    print("\n[4/6] Analyzing User vs Movie features...")
    analyze_user_vs_movie_features(X, y, feature_groups)

    # Detect problems
    print("\n[5/6] Detecting problematic features...")
    issues = detect_problematic_features(X, y, feature_groups, dimension_metadata)

    # Generate plots
    print("\n[6/6] Generating visualizations...")
    output_dir = BASE_DIR / "outputs" / "linear_explainable" / "feature_analysis"
    plot_feature_distributions(X, feature_groups, output_dir)

    # Save detailed report
    print("\n" + "=" * 80)
    print("SAVING DETAILED REPORT")
    print("=" * 80)

    report = {
        'feature_groups': all_stats,
        'issues': issues,
        'summary': {
            'total_features': int(X.shape[1]),
            'n_groups': len(feature_groups['groups']),
            'n_issues': len(issues)
        }
    }

    report_path = output_dir / "feature_analysis_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✅ Detailed report saved: {report_path}")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Find max scale difference
    scales = [s['stats']['typical_scale'] for s in all_stats]
    max_scale = max(scales)
    min_scale = min([s for s in scales if s > 0])
    ratio = max_scale / min_scale

    if ratio > 100:
        print("\n⚠️  CRITICAL: Feature scales differ by >100x")
        print("   → Recommend: Standardize all features to same scale")
        print("   → Action: Add feature scaling in 01_prepare_features.py")

    if len([i for i in issues if i['severity'] == 'high']) > 0:
        print("\n⚠️  HIGH: Some features are constant or near-zero")
        print("   → Recommend: Remove or fix these features")

    # Check user features
    user_stats = [s for s in all_stats if s['group'].startswith('user_')]
    user_mean_corr = np.mean([s['correlations']['mean_abs_corr'] for s in user_stats])

    movie_stats = [s for s in all_stats if s['group'].startswith('movie_')]
    movie_mean_corr = np.mean([s['correlations']['mean_abs_corr'] for s in movie_stats])

    if movie_mean_corr > user_mean_corr * 2:
        print("\n⚠️  Movie features have much higher correlation than user features")
        print("   → This explains why model ignores user features")
        print("   → Action: Consider boosting user feature scales")

    print("\n" + "=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    main()