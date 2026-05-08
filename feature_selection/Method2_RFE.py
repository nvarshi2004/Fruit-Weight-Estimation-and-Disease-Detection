#!/usr/bin/env python3
"""
Method 2: Recursive Feature Elimination (RFE) with Complete ML Pipeline
Implements wrapper-based feature selection using RFE algorithm
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import RFE
import xgboost as xgb

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_directory_structure(base_path, stages):
    """Create organized folder structure for outputs"""
    splits = ['80-20', '70-30', '60-40']
    
    for stage in stages:
        stage_path = os.path.join(base_path, stage)
        for split in splits:
            split_path = os.path.join(stage_path, split)
            os.makedirs(split_path, exist_ok=True)
    
    print(f"✓ Directory structure created at: {base_path}")

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if not mask.any():
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def clean_data(df):
    """Handle missing values, infinite values, and non-numeric columns"""
    print("\n" + "="*70)
    print("DATA CLEANING")
    print("="*70)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"⚠ Missing values found:\n{missing[missing > 0]}")
        df = df.fillna(df.mean(numeric_only=True))
        print("✓ Missing values filled with column means")
    
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))
    
    # Ensure numeric columns
    for col in df.columns[2:]:  # Skip ID and target
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(df.mean(numeric_only=True))
    print(f"✓ Data cleaning completed. Shape: {df.shape}")
    
    return df

def get_ml_models():
    """Return dictionary of all ML models"""
    return {
        'Linear_Regression': LinearRegression(),
        'Lasso': Lasso(alpha=1.0, random_state=42),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient_Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
        'SVR': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'MLP_Regressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Train model and return comprehensive metrics"""
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_test_pred)
    mape = calculate_mape(y_test, y_test_pred)
    overfit = r2_train - r2_test
    
    metrics = {
        'R2_Test': r2_test,
        'R2_Train': r2_train,
        'Overfit': overfit,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return metrics, y_test_pred

def save_predictions(y_test, y_pred, test_indices, output_path, model_name):
    """Save prediction results to CSV"""
    pred_df = pd.DataFrame({
        'Index': test_indices,
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Error': y_test.values - y_pred,
        'Abs_Error': np.abs(y_test.values - y_pred),
        'Pct_Error': np.abs((y_test.values - y_pred) / y_test.values) * 100
    })
    pred_df.to_csv(output_path, index=False)

def create_scatter_plot(y_test, y_pred, r2, model_name, output_path):
    """Create actual vs predicted scatter plot"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
    plt.title(f'{model_name}\nActual vs Predicted (R² = {r2:.4f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plot(y_test, y_pred, model_name, output_path):
    """Create residual plot"""
    residuals = y_test.values - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=50)
    axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[0].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Residual distribution
    axes[1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_models_for_split(X_train, X_test, y_train, y_test, test_indices, output_dir, split_name, stage_name):
    """Run all models for a specific train-test split"""
    print(f"\n  Processing {split_name} split...")
    
    models = get_ml_models()
    results = []
    
    for model_name, model in models.items():
        try:
            # Evaluate model
            metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test)
            
            # Save predictions
            pred_path = os.path.join(output_dir, f"{model_name}_Predictions.csv")
            save_predictions(y_test, y_pred, test_indices, pred_path, model_name)
            
            # Create scatter plot
            scatter_path = os.path.join(output_dir, f"{model_name}_Scatter.png")
            create_scatter_plot(y_test, y_pred, metrics['R2_Test'], model_name, scatter_path)
            
            # Create residual plot
            residual_path = os.path.join(output_dir, f"{model_name}_Residual.png")
            create_residual_plot(y_test, y_pred, model_name, residual_path)
            
            # Store results
            result = {
                'Stage': stage_name,
                'Split': split_name,
                'Model': model_name,
                **metrics
            }
            results.append(result)
            
            print(f"    ✓ {model_name}: R² = {metrics['R2_Test']:.4f}")
            
        except Exception as e:
            print(f"    ✗ {model_name} failed: {str(e)}")
    
    # Save model results summary
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "Model_Results.csv")
    results_df.to_csv(results_path, index=False)
    
    return results

def create_comprehensive_analysis(overall_results, output_path, stage_names):
    """Create comprehensive multi-panel comparison visualization"""
    df = pd.DataFrame(overall_results)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. R² comparison across all configurations
    ax1 = fig.add_subplot(gs[0, :])
    pivot_r2 = df.pivot_table(values='R2_Test', index='Model', columns=['Stage', 'Split'], aggfunc='mean')
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, ax=ax1, cbar_kws={'label': 'R²'})
    ax1.set_title('R² Test Score Heatmap (All Configurations)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('')
    ax1.set_ylabel('Model', fontsize=11, fontweight='bold')
    
    # 2. RMSE comparison
    ax2 = fig.add_subplot(gs[1, 0])
    for stage in stage_names:
        stage_data = df[df['Stage'] == stage]
        stage_data.groupby('Model')['RMSE'].mean().plot(kind='barh', ax=ax2, alpha=0.7, label=stage)
    ax2.set_xlabel('RMSE', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Model', fontsize=10, fontweight='bold')
    ax2.set_title('Average RMSE by Model', fontsize=12, fontweight='bold')
    ax2.legend(title='Stage', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. MAE comparison
    ax3 = fig.add_subplot(gs[1, 1])
    for stage in stage_names:
        stage_data = df[df['Stage'] == stage]
        stage_data.groupby('Model')['MAE'].mean().plot(kind='barh', ax=ax3, alpha=0.7, label=stage)
    ax3.set_xlabel('MAE', fontsize=10, fontweight='bold')
    ax3.set_ylabel('')
    ax3.set_title('Average MAE by Model', fontsize=12, fontweight='bold')
    ax3.legend(title='Stage', fontsize=8)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # 4. MAPE comparison
    ax4 = fig.add_subplot(gs[1, 2])
    for stage in stage_names:
        stage_data = df[df['Stage'] == stage]
        mape_data = stage_data.groupby('Model')['MAPE'].mean()
        mape_data = mape_data[mape_data < 100]
        mape_data.plot(kind='barh', ax=ax4, alpha=0.7, label=stage)
    ax4.set_xlabel('MAPE (%)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('')
    ax4.set_title('Average MAPE by Model', fontsize=12, fontweight='bold')
    ax4.legend(title='Stage', fontsize=8)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Overfitting analysis
    ax5 = fig.add_subplot(gs[2, 0])
    overfit_data = df.groupby('Model')['Overfit'].mean().sort_values()
    colors = ['green' if x < 0.1 else 'orange' if x < 0.2 else 'red' for x in overfit_data.values]
    overfit_data.plot(kind='barh', ax=ax5, color=colors, alpha=0.7)
    ax5.axvline(x=0.1, color='orange', linestyle='--', label='Moderate (0.1)', alpha=0.7)
    ax5.axvline(x=0.2, color='red', linestyle='--', label='High (0.2)', alpha=0.7)
    ax5.set_xlabel('Overfit (Train R² - Test R²)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Model', fontsize=10, fontweight='bold')
    ax5.set_title('Overfitting Analysis', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 6. R² by split comparison
    ax6 = fig.add_subplot(gs[2, 1])
    split_comparison = df.groupby(['Split', 'Model'])['R2_Test'].mean().unstack()
    split_comparison.plot(kind='bar', ax=ax6, width=0.8)
    ax6.set_xlabel('Train-Test Split', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Average R²', fontsize=10, fontweight='bold')
    ax6.set_title('R² by Split Ratio', fontsize=12, fontweight='bold')
    ax6.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=0)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Best model per stage
    ax7 = fig.add_subplot(gs[2, 2])
    best_models = df.groupby('Stage').apply(lambda x: x.loc[x['R2_Test'].idxmax(), 'Model'])
    best_r2 = df.groupby('Stage')['R2_Test'].max()
    best_data = pd.DataFrame({'Model': best_models.values, 'R2': best_r2.values}, index=best_models.index)
    
    x_pos = np.arange(len(best_data))
    bars = ax7.bar(x_pos, best_data['R2'], alpha=0.7, edgecolor='black')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(best_data.index, rotation=45, ha='right')
    ax7.set_ylabel('Best R²', fontsize=10, fontweight='bold')
    ax7.set_title('Best Model per Stage', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Add model names on bars
    for i, (bar, model) in enumerate(zip(bars, best_data['Model'])):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{model}\n{height:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.suptitle('Comprehensive Model Performance Analysis - RFE Method', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Comprehensive analysis saved: {output_path}")

def create_lr_comparison(overall_results, output_path):
    """Create Linear Regression specific comparison and R² visualization"""
    df = pd.DataFrame(overall_results)
    lr_data = df[df['Model'] == 'Linear_Regression'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. R² comparison across stages and splits
    ax1 = axes[0, 0]
    pivot_r2 = lr_data.pivot_table(values='R2_Test', index='Stage', columns='Split')
    pivot_r2.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_xlabel('Stage', fontsize=11, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax1.set_title('Linear Regression: R² Across Stages and Splits', fontsize=13, fontweight='bold')
    ax1.legend(title='Split', fontsize=9)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    
    # 2. All metrics comparison
    ax2 = axes[0, 1]
    metrics_avg = lr_data.groupby('Stage')[['R2_Test', 'RMSE', 'MAE']].mean()
    
    # Normalize metrics for comparison
    metrics_norm = metrics_avg.copy()
    metrics_norm['R2_Test'] = metrics_norm['R2_Test']
    metrics_norm['RMSE'] = metrics_norm['RMSE'] / metrics_norm['RMSE'].max()
    metrics_norm['MAE'] = metrics_norm['MAE'] / metrics_norm['MAE'].max()
    
    x = np.arange(len(metrics_norm.index))
    width = 0.25
    
    ax2.bar(x - width, metrics_norm['R2_Test'], width, label='R² Test', alpha=0.8)
    ax2.bar(x, metrics_norm['RMSE'], width, label='RMSE (norm)', alpha=0.8)
    ax2.bar(x + width, metrics_norm['MAE'], width, label='MAE (norm)', alpha=0.8)
    
    ax2.set_xlabel('Stage', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score (Normalized)', fontsize=11, fontweight='bold')
    ax2.set_title('Linear Regression: Metrics Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_norm.index, rotation=45, ha='right')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. R² Train vs Test (Overfitting)
    ax3 = axes[1, 0]
    overfit_data = lr_data.groupby('Stage')[['R2_Train', 'R2_Test']].mean()
    overfit_data.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Stage', fontsize=11, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Regression: Train vs Test R² (Overfitting Check)', fontsize=13, fontweight='bold')
    ax3.legend(['R² Train', 'R² Test'], fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Error metrics heatmap
    ax4 = axes[1, 1]
    rmse_pivot = lr_data.pivot_table(values='RMSE', index='Stage', columns='Split')
    sns.heatmap(rmse_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'RMSE'})
    ax4.set_title('Linear Regression: RMSE Heatmap', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Split', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Stage', fontsize=11, fontweight='bold')
    
    plt.suptitle('Linear Regression Performance Analysis - RFE Method', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Linear Regression comparison saved: {output_path}")

# ============================================================================
# RFE FEATURE SELECTION
# ============================================================================

def perform_rfe_selection(X, y, feature_names):
    """Perform RFE feature selection with multiple stages"""
    print("\n" + "="*70)
    print("RFE FEATURE SELECTION")
    print("="*70)
    
    try:
        # Use Random Forest as base estimator
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Different stages: select top 80%, 60%, 40% of features
        n_features = len(feature_names)
        stages = {
            'RFE_Top80%': max(1, int(n_features * 0.8)),
            'RFE_Top60%': max(1, int(n_features * 0.6)),
            'RFE_Top40%': max(1, int(n_features * 0.4))
        }
        
        all_results = {}
        
        for stage_name, n_features_to_select in stages.items():
            print(f"\nRunning RFE for {stage_name} (selecting {n_features_to_select}/{n_features} features)...")
            
            rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
            rfe.fit(X.values, y.values)
            
            # Get selected features
            selected_mask = rfe.support_
            selected_features = feature_names[selected_mask]
            rankings = rfe.ranking_
            
            # Store results
            results_df = pd.DataFrame({
                'Feature': feature_names,
                'Selected': selected_mask,
                'Ranking': rankings,
                'Decision': [f'Rank {r}' for r in rankings]
            })
            results_df = results_df.sort_values('Ranking')
            
            all_results[stage_name] = {
                'features': selected_features,
                'results_df': results_df,
                'selector': rfe
            }
            
            print(f"✓ {stage_name} completed: {len(selected_features)} features selected")
            print(f"  Selected: {', '.join(selected_features)}")
        
        return all_results
        
    except Exception as e:
        print(f"✗ RFE failed: {str(e)}")
        print("  Using all features as fallback")
        results_df = pd.DataFrame({
            'Feature': feature_names,
            'Selected': True,
            'Ranking': 1,
            'Decision': 'All Features (Fallback)'
        })
        return {
            'Before_RFE': {
                'features': feature_names,
                'results_df': results_df,
                'selector': None
            }
        }

def visualize_rfe_results(all_results, output_path):
    """Create visualization of RFE results"""
    n_stages = len(all_results)
    fig, axes = plt.subplots(1, n_stages, figsize=(8*n_stages, 6))
    
    if n_stages == 1:
        axes = [axes]
    
    for idx, (stage_name, result_data) in enumerate(all_results.items()):
        ax = axes[idx]
        results_df = result_data['results_df']
        
        # Feature rankings bar chart
        colors = ['green' if r == 1 else 'orange' if r <= 3 else 'red' for r in results_df['Ranking']]
        ax.barh(results_df['Feature'], results_df['Ranking'], color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Ranking (Lower is Better)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Features', fontsize=11, fontweight='bold')
        ax.set_title(f'{stage_name}\n({len(result_data["features"])} features)', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('RFE Feature Rankings - Multiple Stages', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ RFE visualization saved: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("RFE FEATURE SELECTION & ML PIPELINE")
    print("="*70)
    
    # Get input file path
    print("\nPlease enter the path to your Excel file:")
    file_path = input("File path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        sys.exit(1)
    
    # Get output directory
    print("\nPlease enter the output directory path:")
    output_base = input("Output directory: ").strip().strip('"').strip("'")
    
    try:
        # Load data
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        df = pd.read_excel(file_path)
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Clean data
        df = clean_data(df)
        
        # Extract components
        sample_ids = df.iloc[:, 0]
        y = df.iloc[:, 1]
        X = df.iloc[:, 2:]
        feature_names = np.array(X.columns)
        
        print(f"\n✓ Data structure:")
        print(f"  Sample IDs: {sample_ids.name}")
        print(f"  Target: {y.name}")
        print(f"  Features: {len(feature_names)} features")
        
        # Perform RFE selection
        rfe_results = perform_rfe_selection(X, y, feature_names)
        
        # Create directory structure with all stages
        base_path = os.path.join(output_base, "Method2_RFE")
        stages = ['Before_RFE'] + list(rfe_results.keys())
        create_directory_structure(base_path, stages)
        
        # Save RFE results
        for stage_name, result_data in rfe_results.items():
            report_path = os.path.join(base_path, f"{stage_name}_Selection_Report.csv")
            result_data['results_df'].to_csv(report_path, index=False)
            print(f"✓ {stage_name} report saved: {report_path}")
        
        # Visualize RFE results
        rfe_viz_path = os.path.join(base_path, "RFE_Rankings.png")
        visualize_rfe_results(rfe_results, rfe_viz_path)
        
        # Define train-test splits
        splits = {
            '80-20': 0.2,
            '70-30': 0.3,
            '60-40': 0.4
        }
        
        # Store all results
        overall_results = []
        
        # Process each stage (Before + RFE stages)
        stage_configs = [('Before_RFE', feature_names)]
        for stage_name, result_data in rfe_results.items():
            stage_configs.append((stage_name, result_data['features']))
        
        for stage_name, features_to_use in stage_configs:
            print("\n" + "="*70)
            print(f"STAGE: {stage_name}")
            print(f"Features: {len(features_to_use)}")
            print("="*70)
            
            X_stage = X[features_to_use]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_stage)
            X_scaled = pd.DataFrame(X_scaled, columns=features_to_use, index=X_stage.index)
            
            # Process each split
            for split_name, test_size in splits.items():
                print(f"\n{'─'*70}")
                print(f"Processing {split_name} split ({int((1-test_size)*100)}% train, {int(test_size*100)}% test)")
                print(f"{'─'*70}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42
                )
                
                # Get test indices for tracking
                test_indices = X_test.index.tolist()
                
                # Create output directory
                output_dir = os.path.join(base_path, stage_name, split_name)
                
                # Run all models
                split_results = run_models_for_split(
                    X_train, X_test, y_train, y_test, 
                    test_indices, output_dir, split_name, stage_name
                )
                
                overall_results.extend(split_results)
        
        # Save overall summary
        print("\n" + "="*70)
        print("SAVING OVERALL SUMMARY")
        print("="*70)
        
        overall_df = pd.DataFrame(overall_results)
        overall_path = os.path.join(base_path, "Overall_Summary.csv")
        overall_df.to_csv(overall_path, index=False)
        print(f"✓ Overall summary saved: {overall_path}")
        
        # Create comprehensive analysis
        comp_analysis_path = os.path.join(base_path, "Comprehensive_Analysis.png")
        create_comprehensive_analysis(overall_results, comp_analysis_path, stages)
        
        # Create Linear Regression comparison
        lr_comparison_path = os.path.join(base_path, "Linear_Regression_Comparison.png")
        create_lr_comparison(overall_results, lr_comparison_path)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        summary_stats = overall_df.groupby(['Stage', 'Model']).agg({
            'R2_Test': 'mean',
            'RMSE': 'mean',
            'MAE': 'mean'
        }).round(4)
        
        print("\nAverage Performance by Stage and Model:")
        print(summary_stats.to_string())
        
        best_overall = overall_df.loc[overall_df['R2_Test'].idxmax()]
        print(f"\n🏆 Best Overall Configuration:")
        print(f"   Stage: {best_overall['Stage']}")
        print(f"   Model: {best_overall['Model']}")
        print(f"   Split: {best_overall['Split']}")
        print(f"   R² Test: {best_overall['R2_Test']:.4f}")
        print(f"   RMSE: {best_overall['RMSE']:.4f}")
        
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"✓ All results saved to: {base_path}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
