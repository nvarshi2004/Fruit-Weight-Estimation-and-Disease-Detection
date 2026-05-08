#!/usr/bin/env python3
"""
Method 5: Random Forest Feature Importance & Complete ML Pipeline
Implements embedded feature selection using Random Forest importance scores
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
    ax7.set_xticklabels(best_data.index, rotation=45, ha='right', fontsize=7)
    ax7.set_ylabel('Best R²', fontsize=10, fontweight='bold')
    ax7.set_title('Best Model per Stage', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, model) in enumerate(zip(bars, best_data['Model'])):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{model}\n{height:.3f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.suptitle('Comprehensive Model Performance Analysis - RF Importance Method', 
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
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate (0.5)')
    
    # 2. All metrics comparison
    ax2 = axes[0, 1]
    metrics_avg = lr_data.groupby('Stage')[['R2_Test', 'RMSE', 'MAE']].mean()
    
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
    ax2.set_xticklabels(metrics_norm.index, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. R² Train vs Test
    ax3 = axes[1, 0]
    overfit_data = lr_data.groupby('Stage')[['R2_Train', 'R2_Test']].mean()
    overfit_data.plot(kind='bar', ax=ax3, width=0.8)
    ax3.set_xlabel('Stage', fontsize=11, fontweight='bold')
    ax3.set_ylabel('R² Score', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Regression: Train vs Test R²', fontsize=13, fontweight='bold')
    ax3.legend(['R² Train', 'R² Test'], fontsize=9)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. RMSE heatmap
    ax4 = axes[1, 1]
    rmse_pivot = lr_data.pivot_table(values='RMSE', index='Stage', columns='Split')
    sns.heatmap(rmse_pivot, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax4, cbar_kws={'label': 'RMSE'})
    ax4.set_title('Linear Regression: RMSE Heatmap', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Split', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Stage', fontsize=11, fontweight='bold')
    
    plt.suptitle('Linear Regression Performance Analysis - RF Importance Method', 
                 fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Linear Regression comparison saved: {output_path}")

# ============================================================================
# RANDOM FOREST IMPORTANCE
# ============================================================================

def perform_rf_importance_selection(X, y, feature_names):
    """Perform RF importance-based feature selection with different thresholds"""
    print("\n" + "="*70)
    print("RANDOM FOREST FEATURE IMPORTANCE SELECTION")
    print("="*70)
    
    try:
        # Train Random Forest to get importance scores
        print("\nTraining Random Forest to calculate feature importances...")
        rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
        rf.fit(X.values, y.values)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Different threshold strategies
        mean_importance = np.mean(importances)
        median_importance = np.median(importances)
        
        thresholds = {
            'RF_Top80%': np.percentile(importances, 20),  # Top 80%
            'RF_Top60%': np.percentile(importances, 40),  # Top 60%
            'RF_AboveMean': mean_importance  # Above mean
        }
        
        all_results = {}
        
        for stage_name, threshold in thresholds.items():
            print(f"\nApplying threshold for {stage_name} (threshold={threshold:.4f})...")
            
            # Select features above threshold
            selected_mask = importances > threshold
            selected_features = feature_names[selected_mask]
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Selected': selected_mask,
                'Decision': ['Selected' if s else 'Rejected' for s in selected_mask]
            })
            results_df = results_df.sort_values('Importance', ascending=False)
            
            all_results[stage_name] = {
                'features': selected_features,
                'results_df': results_df,
                'selector': rf,
                'threshold': threshold
            }
            
            print(f"✓ {stage_name} completed: {len(selected_features)}/{len(feature_names)} features selected")
            print(f"  Top 3 features: {', '.join(results_df.head(3)['Feature'].values)}")
        
        return all_results, importances
        
    except Exception as e:
        print(f"✗ RF Importance failed: {str(e)}")
        print("  Using all features as fallback")
        results_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': 1.0 / len(feature_names),
            'Selected': True,
            'Decision': 'All Features (Fallback)'
        })
        return {
            'Before_RF': {
                'features': feature_names,
                'results_df': results_df,
                'selector': None,
                'threshold': 0
            }
        }, np.ones(len(feature_names)) / len(feature_names)

def visualize_rf_importance_results(all_results, importances, feature_names, output_path):
    """Create visualization of RF importance results"""
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall feature importance bar chart
    ax1 = fig.add_subplot(gs[0, :])
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importances = importances[sorted_idx]
    
    colors_main = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    ax1.barh(sorted_features, sorted_importances, color=colors_main, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
    ax1.set_title('Random Forest Feature Importance Scores', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add mean and median lines
    mean_imp = np.mean(importances)
    median_imp = np.median(importances)
    ax1.axvline(x=mean_imp, color='red', linestyle='--', linewidth=2, label=f'Mean ({mean_imp:.4f})')
    ax1.axvline(x=median_imp, color='orange', linestyle='--', linewidth=2, label=f'Median ({median_imp:.4f})')
    ax1.legend(fontsize=10)
    
    # 2-4. Feature selection for each threshold
    positions = [(1, 0), (1, 1), (1, 2)]
    for idx, (stage_name, result_data) in enumerate(all_results.items()):
        if idx >= len(positions):
            break
            
        ax = fig.add_subplot(gs[positions[idx]])
        results_df = result_data['results_df']
        threshold = result_data['threshold']
        
        colors = ['green' if s else 'lightgray' for s in results_df['Selected']]
        ax.barh(results_df['Feature'], results_df['Importance'], color=colors, alpha=0.7, edgecolor='black')
        ax.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold')
        ax.set_xlabel('Importance', fontsize=10, fontweight='bold')
        ax.set_ylabel('Features', fontsize=10, fontweight='bold')
        ax.set_title(f'{stage_name}\n({len(result_data["features"])} selected)', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    
    plt.suptitle('Random Forest Feature Importance Selection Results', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ RF Importance visualization saved: {output_path}")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("RANDOM FOREST IMPORTANCE FEATURE SELECTION & ML PIPELINE")
    print("="*70)
    
    # Get input
    print("\nPlease enter the path to your Excel file:")
    file_path = input("File path: ").strip().strip('"').strip("'")
    
    if not os.path.exists(file_path):
        print(f"✗ Error: File not found: {file_path}")
        sys.exit(1)
    
    print("\nPlease enter the output directory path:")
    output_base = input("Output directory: ").strip().strip('"').strip("'")
    
    try:
        # Load data
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        df = pd.read_excel(file_path)
        print(f"✓ Data loaded. Shape: {df.shape}")
        
        # Clean data
        df = clean_data(df)
        
        # Extract components
        sample_ids = df.iloc[:, 0]
        y = df.iloc[:, 1]
        X = df.iloc[:, 2:]
        feature_names = np.array(X.columns)
        
        print(f"\n✓ Data structure:")
        print(f"  Target: {y.name}")
        print(f"  Features: {len(feature_names)}")
        
        # Perform RF Importance selection
        rf_results, importances = perform_rf_importance_selection(X, y, feature_names)
        
        # Create directory structure
        base_path = os.path.join(output_base, "Method5_RF_Importance")
        stages = ['Before_RF'] + list(rf_results.keys())
        create_directory_structure(base_path, stages)
        
        # Save results
        for stage_name, result_data in rf_results.items():
            report_path = os.path.join(base_path, f"{stage_name}_Selection_Report.csv")
            result_data['results_df'].to_csv(report_path, index=False)
            print(f"✓ {stage_name} report saved")
        
        # Visualize
        viz_path = os.path.join(base_path, "RF_Importance_Rankings.png")
        visualize_rf_importance_results(rf_results, importances, feature_names, viz_path)
        
        # Define splits
        splits = {'80-20': 0.2, '70-30': 0.3, '60-40': 0.4}
        
        # Store results
        overall_results = []
        
        # Process stages
        stage_configs = [('Before_RF', feature_names)]
        for stage_name, result_data in rf_results.items():
            stage_configs.append((stage_name, result_data['features']))
        
        for stage_name, features_to_use in stage_configs:
            print("\n" + "="*70)
            print(f"STAGE: {stage_name} ({len(features_to_use)} features)")
            print("="*70)
            
            X_stage = X[features_to_use]
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X_stage), 
                                   columns=features_to_use, index=X_stage.index)
            
            for split_name, test_size in splits.items():
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=42)
                
                output_dir = os.path.join(base_path, stage_name, split_name)
                split_results = run_models_for_split(
                    X_train, X_test, y_train, y_test, 
                    X_test.index.tolist(), output_dir, split_name, stage_name)
                
                overall_results.extend(split_results)
        
        # Save summary
        print("\n" + "="*70)
        print("SAVING SUMMARY")
        print("="*70)
        
        overall_df = pd.DataFrame(overall_results)
        overall_df.to_csv(os.path.join(base_path, "Overall_Summary.csv"), index=False)
        
        # Create visualizations
        create_comprehensive_analysis(overall_results, 
            os.path.join(base_path, "Comprehensive_Analysis.png"), stages)
        create_lr_comparison(overall_results, 
            os.path.join(base_path, "Linear_Regression_Comparison.png"))
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        summary_stats = overall_df.groupby(['Stage', 'Model']).agg({
            'R2_Test': 'mean', 'RMSE': 'mean', 'MAE': 'mean'}).round(4)
        print("\n" + summary_stats.to_string())
        
        best = overall_df.loc[overall_df['R2_Test'].idxmax()]
        print(f"\n🏆 Best: {best['Model']} | {best['Stage']} | {best['Split']}")
        print(f"   R²={best['R2_Test']:.4f} | RMSE={best['RMSE']:.4f}")
        
        print("\n" + "="*70)
        print(f"✓ COMPLETED! Results: {base_path}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
