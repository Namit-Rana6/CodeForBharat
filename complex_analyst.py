    #!/usr/bin/env python3
    """
    High-Quality Multi-Agent Data Analyst with actual code execution and statistical analysis.
    """
    import os
    import sys

    # Fix Windows console encoding
    if sys.platform.startswith('win'):
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import warnings
    warnings.filterwarnings('ignore')
    from langchain_google_genai import ChatGoogleGenerativeAI
    from typing import Dict, List
    import io

    class DataAnalysisAgent:
        """Advanced data analysis agent that executes real code and statistical analysis."""
        
        def __init__(self, role: str, goal: str, backstory: str, llm):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.llm = llm
            self.memory = []
            self.df = None
            self.analysis_results = {}
        
        def set_dataframe(self, df: pd.DataFrame):
            """Set the dataframe for analysis."""
            self.df = df.copy()
        
        def execute_code_analysis(self, code_description: str) -> Dict:
            """Execute actual statistical analysis and return results."""
            results = {}
            
            if self.df is None:
                return {"error": "No dataframe set for analysis"}
            
            try:
                # Perform actual statistical analysis based on role
                if "Discovery" in self.role or "Profiling" in self.role:
                    results = self._discovery_analysis()
                elif "Quality" in self.role or "Preparation" in self.role:
                    results = self._structure_analysis()
                elif "Statistical" in self.role:
                    results = self._eda_analysis()
                elif "Visualization" in self.role:
                    results = self._visualization_analysis()
                elif "Insights" in self.role or "Reporting" in self.role:
                    results = self._business_analysis()
                elif "Auto" in self.role:
                    results = self._auto_analysis()
                    
                self.analysis_results = results
                return results
                
            except Exception as e:
                return {"error": f"Analysis failed: {str(e)}"}
        
        def _structure_analysis(self) -> Dict:
            """Perform comprehensive data structure analysis."""
            results = {
                "basic_info": {
                    "shape": self.df.shape,
                    "columns": list(self.df.columns),
                    "dtypes": self.df.dtypes.to_dict(),
                    "memory_usage": self.df.memory_usage(deep=True).sum(),
                    "duplicate_rows": self.df.duplicated().sum()
                },
                "missing_analysis": {
                    "missing_counts": self.df.isnull().sum().to_dict(),
                    "missing_percentages": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
                    "complete_rows": len(self.df) - self.df.isnull().any(axis=1).sum()
                },
                "data_quality": {},
                "recommendations": []
            }
            
            # Data quality assessment
            total_cells = self.df.shape[0] * self.df.shape[1]
            missing_cells = self.df.isnull().sum().sum()
            completeness_score = (total_cells - missing_cells) / total_cells
            
            # Assess data type appropriateness
            type_issues = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    # Check if should be categorical
                    unique_ratio = self.df[col].nunique() / len(self.df)
                    if unique_ratio < 0.05:
                        type_issues.append(f"{col}: Should be categorical (only {self.df[col].nunique()} unique values)")
            
            results["data_quality"] = {
                "completeness_score": completeness_score,
                "quality_grade": "A" if completeness_score > 0.95 else "B" if completeness_score > 0.8 else "C",
                "type_issues": type_issues
            }
            
            # Generate recommendations
            if missing_cells > 0:
                results["recommendations"].append("Implement missing data imputation strategy")
            if len(type_issues) > 0:
                results["recommendations"].append("Convert appropriate columns to categorical type")
            if self.df.duplicated().sum() > 0:
                results["recommendations"].append("Remove duplicate rows")
                
            return results
        
        def _eda_analysis(self) -> Dict:
            """Perform comprehensive exploratory data analysis."""
            results = {
                "numerical_analysis": {},
                "categorical_analysis": {},
                "correlation_analysis": {},
                "distribution_analysis": {},
                "outlier_analysis": {},
                "statistical_tests": {}
            }
            
            # Numerical analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                results["numerical_analysis"] = {
                    "descriptive_stats": self.df[numeric_cols].describe().to_dict(),
                    "skewness": self.df[numeric_cols].skew().to_dict(),
                    "kurtosis": self.df[numeric_cols].kurtosis().to_dict()
                }
                
                # Correlation analysis
                corr_matrix = self.df[numeric_cols].corr()
                results["correlation_analysis"] = {
                    "correlation_matrix": corr_matrix.to_dict(),
                    "strong_correlations": []
                }
                
                # Find strong correlations
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            results["correlation_analysis"]["strong_correlations"].append({
                                "var1": corr_matrix.columns[i],
                                "var2": corr_matrix.columns[j], 
                                "correlation": corr_val
                            })
            
            # Categorical analysis
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_analysis = {}
                for col in categorical_cols:
                    cat_analysis[col] = {
                        "unique_count": self.df[col].nunique(),
                        "most_frequent": self.df[col].value_counts().head().to_dict(),
                        "entropy": stats.entropy(self.df[col].value_counts())
                    }
                results["categorical_analysis"] = cat_analysis
            
            # Outlier detection using IQR method
            outlier_analysis = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                outlier_analysis[col] = {
                    "outlier_count": len(outliers),
                    "outlier_percentage": len(outliers) / len(self.df) * 100,
                    "bounds": {"lower": lower_bound, "upper": upper_bound}
                }
            results["outlier_analysis"] = outlier_analysis
            
            # Statistical tests
            if 'Survived' in self.df.columns:
                # Chi-square tests for categorical variables
                for col in categorical_cols:
                    if col != 'Survived':
                        try:
                            contingency_table = pd.crosstab(self.df[col], self.df['Survived'])
                            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                            results["statistical_tests"][f"{col}_vs_survived"] = {
                                "test": "chi-square",
                                "chi2_statistic": chi2,
                                "p_value": p_value,
                                "significant": p_value < 0.05
                            }
                        except:
                            pass
            
            return results
        
        def _business_analysis(self) -> Dict:
            """Perform business-focused analysis with actionable insights."""
            results = {
                "key_insights": [],
                "business_metrics": {},
                "segment_analysis": {},
                "recommendations": []
            }
            
            # Generate insights based on the data
            if 'Survived' in self.df.columns:
                survival_rate = self.df['Survived'].mean()
                results["business_metrics"]["overall_survival_rate"] = survival_rate
                
                # Gender analysis
                if 'Sex' in self.df.columns:
                    gender_survival = self.df.groupby('Sex')['Survived'].agg(['mean', 'count']).to_dict()
                    results["segment_analysis"]["gender"] = gender_survival
                    
                    female_survival = self.df[self.df['Sex'] == 'female']['Survived'].mean()
                    male_survival = self.df[self.df['Sex'] == 'male']['Survived'].mean()
                    
                    results["key_insights"].append({
                        "insight": "Gender Survival Gap",
                        "description": f"Female survival rate ({female_survival:.1%}) is {female_survival/male_survival:.1f}x higher than male rate ({male_survival:.1%})",
                        "business_impact": "High",
                        "actionable": "Prioritize safety protocols for male passengers"
                    })
                
                # Class analysis
                if 'Pclass' in self.df.columns:
                    class_survival = self.df.groupby('Pclass')['Survived'].agg(['mean', 'count']).to_dict()
                    results["segment_analysis"]["passenger_class"] = class_survival
                    
                    class_1_survival = self.df[self.df['Pclass'] == 1]['Survived'].mean()
                    class_3_survival = self.df[self.df['Pclass'] == 3]['Survived'].mean()
                    
                    results["key_insights"].append({
                        "insight": "Class-Based Survival Disparity", 
                        "description": f"First class survival ({class_1_survival:.1%}) vs Third class ({class_3_survival:.1%})",
                        "business_impact": "Critical",
                        "actionable": "Ensure equitable safety access across all classes"
                    })
                
                # Age analysis
                if 'Age' in self.df.columns:
                    age_bins = pd.cut(self.df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
                    age_survival = self.df.groupby(age_bins)['Survived'].mean().to_dict()
                    results["segment_analysis"]["age_groups"] = age_survival
            
            # Economic analysis
            if 'Fare' in self.df.columns:
                fare_stats = self.df['Fare'].describe().to_dict()
                results["business_metrics"]["fare_analysis"] = fare_stats
                
                # Revenue insights
                total_revenue = self.df['Fare'].sum()
                avg_fare = self.df['Fare'].mean()
                results["business_metrics"]["revenue_metrics"] = {
                    "total_revenue": total_revenue,
                    "average_fare": avg_fare,
                    "revenue_per_survivor": total_revenue / self.df['Survived'].sum() if 'Survived' in self.df.columns else None
                }
            
            return results
        
        def _business_analysis(self) -> Dict:
            """Perform objective-focused analysis with actionable insights."""
            results = {
                "key_insights": [],
                "summary_statistics": {},
                "data_story": "",
                "actionable_recommendations": []
            }
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Generate summary statistics
            results["summary_statistics"] = {
                "total_records": len(self.df),
                "numeric_variables": len(numeric_cols),
                "categorical_variables": len(categorical_cols),
                "missing_data_percentage": (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
                "complete_records": len(self.df.dropna())
            }
            
            # Generate key insights based on data
            insights = []
            
            # Analyze numeric variables
            for col in numeric_cols:
                if self.df[col].std() > 0:  # Has variation
                    insights.append({
                        "type": "distribution",
                        "variable": col,
                        "insight": f"{col}: Mean = {self.df[col].mean():.2f}, Std = {self.df[col].std():.2f}",
                        "interpretation": f"Variable shows {'high' if self.df[col].std() > self.df[col].mean() else 'moderate'} variability"
                    })
            
            # Analyze categorical variables
            for col in categorical_cols:
                if self.df[col].nunique() <= 10:
                    top_category = self.df[col].value_counts().index[0]
                    top_percentage = (self.df[col].value_counts().iloc[0] / len(self.df)) * 100
                    insights.append({
                        "type": "categorical",
                        "variable": col,
                        "insight": f"{col}: Most common = '{top_category}' ({top_percentage:.1f}%)",
                        "interpretation": f"Distribution is {'highly concentrated' if top_percentage > 50 else 'moderately distributed'}"
                    })
            
            # Find interesting patterns
            for col in self.df.columns:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                if missing_pct > 10:
                    insights.append({
                        "type": "data_quality",
                        "variable": col,
                        "insight": f"{col}: {missing_pct:.1f}% missing data",
                        "interpretation": "Requires attention for data quality"
                    })
            
            results["key_insights"] = insights[:10]  # Top 10 insights
            
            # Generate data story
            story_parts = []
            story_parts.append(f"This dataset contains {len(self.df)} records with {len(self.df.columns)} variables.")
            
            if len(numeric_cols) > 0:
                story_parts.append(f"There are {len(numeric_cols)} numeric variables available for quantitative analysis.")
            
            if len(categorical_cols) > 0:
                story_parts.append(f"There are {len(categorical_cols)} categorical variables for grouping and comparison.")
            
            missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
            if missing_pct > 5:
                story_parts.append(f"Overall missing data is {missing_pct:.1f}%, which requires attention.")
            else:
                story_parts.append("Data quality appears good with minimal missing values.")
            
            results["data_story"] = " ".join(story_parts)
            
            # Generate actionable recommendations
            recommendations = []
            
            if missing_pct > 10:
                recommendations.append("Implement data cleaning strategy to address missing values")
            
            if len(numeric_cols) >= 2:
                recommendations.append("Explore correlations between numeric variables")
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                recommendations.append("Perform group comparisons using categorical variables")
            
            high_var_cols = [col for col in numeric_cols if self.df[col].std() > self.df[col].mean()]
            if high_var_cols:
                recommendations.append(f"Investigate high variability in: {', '.join(high_var_cols[:3])}")
            
            results["actionable_recommendations"] = recommendations
            
            return results
        
        def _auto_analysis(self) -> Dict:
            """Automatically identify key columns and perform intelligent analysis."""
            results = {
                "dataset_profile": {},
                "key_columns_identified": {},
                "intelligent_insights": {},
                "recommended_visualizations": [],
                "data_story": ""
            }
            
            # Analyze dataset characteristics
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Identify target variable automatically
            target_candidates = []
            
            # Look for binary columns (potential targets)
            for col in numeric_cols:
                if self.df[col].nunique() == 2 and set(self.df[col].unique()).issubset({0, 1}):
                    target_candidates.append(col)
            
            # Look for survival/outcome related columns
            outcome_keywords = ['survived', 'outcome', 'target', 'result', 'success', 'fail']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in outcome_keywords):
                    target_candidates.append(col)
            
            primary_target = target_candidates[0] if target_candidates else None
            
            # Identify key categorical predictors
            key_categoricals = []
            for col in categorical_cols:
                unique_ratio = self.df[col].nunique() / len(self.df)
                if 0.01 < unique_ratio < 0.1:  # Good categorical predictors
                    key_categoricals.append(col)
            
            # Identify key numeric predictors
            key_numerics = []
            for col in numeric_cols:
                if col != primary_target and self.df[col].std() > 0:
                    # Check correlation with target if available
                    if primary_target and not self.df[col].isnull().all():
                        corr = abs(self.df[col].corr(self.df[primary_target]))
                        if corr > 0.1:  # Meaningful correlation
                            key_numerics.append((col, corr))
                    else:
                        key_numerics.append((col, self.df[col].std()))
            
            # Sort by importance
            key_numerics = sorted(key_numerics, key=lambda x: x[1], reverse=True)[:5]
            key_numerics = [col[0] for col in key_numerics]
            
            results["key_columns_identified"] = {
                "primary_target": primary_target,
                "key_categorical_predictors": key_categoricals[:5],
                "key_numeric_predictors": key_numerics,
                "total_features": len(self.df.columns)
            }
            
            # Perform intelligent analysis based on identified columns
            insights = {}
            
            # Target analysis
            if primary_target:
                target_dist = self.df[primary_target].value_counts(normalize=True).to_dict()
                insights["target_distribution"] = target_dist
                
                # Analyze relationships with key predictors
                categorical_relationships = {}
                for cat_col in key_categoricals:
                    if cat_col in self.df.columns:
                        crosstab = pd.crosstab(self.df[cat_col], self.df[primary_target], normalize='index')
                        categorical_relationships[cat_col] = crosstab.to_dict()
                
                insights["categorical_relationships"] = categorical_relationships
                
                # Numeric relationships
                numeric_relationships = {}
                for num_col in key_numerics:
                    if num_col in self.df.columns:
                        grouped_stats = self.df.groupby(primary_target)[num_col].agg(['mean', 'median', 'std']).to_dict()
                        numeric_relationships[num_col] = grouped_stats
                
                insights["numeric_relationships"] = numeric_relationships
            
            results["intelligent_insights"] = insights
            
            # Recommend best visualizations
            viz_recommendations = []
            
            if primary_target:
                viz_recommendations.append({
                    "type": "target_distribution",
                    "description": f"Distribution of {primary_target}",
                    "priority": "high",
                    "chart_type": "countplot"
                })
                
                for cat_col in key_categoricals[:3]:
                    viz_recommendations.append({
                        "type": "categorical_relationship",
                        "description": f"{primary_target} by {cat_col}",
                        "priority": "high",
                        "chart_type": "grouped_bar"
                    })
                
                for num_col in key_numerics[:3]:
                    viz_recommendations.append({
                        "type": "numeric_distribution",
                        "description": f"{num_col} distribution by {primary_target}",
                        "priority": "medium",
                        "chart_type": "histogram_by_group"
                    })
            
            # Add correlation heatmap if enough numeric columns
            if len(numeric_cols) >= 3:
                viz_recommendations.append({
                    "type": "correlation_matrix",
                    "description": "Correlation between numeric variables",
                    "priority": "medium",
                    "chart_type": "heatmap"
                })
            
            results["recommended_visualizations"] = viz_recommendations
            
            # Generate data story
            story_parts = []
            if primary_target:
                target_name = primary_target.replace('_', ' ').title()
                story_parts.append(f"This dataset focuses on predicting {target_name}.")
                
                if target_dist := insights.get("target_distribution"):
                    positive_rate = max(target_dist.values()) if target_dist else 0
                    story_parts.append(f"The positive outcome rate is {positive_rate:.1%}.")
                
                if key_categoricals:
                    story_parts.append(f"Key categorical factors include: {', '.join(key_categoricals[:3])}.")
                
                if key_numerics:
                    story_parts.append(f"Important numeric predictors are: {', '.join(key_numerics[:3])}.")
            
            results["data_story"] = " ".join(story_parts)
            
            return results
        
        def _discovery_analysis(self) -> Dict:
            """Discover key patterns and variables in the dataset."""
            results = {
                "dataset_overview": {},
                "variable_types": {},
                "key_patterns": {},
                "recommended_focus": []
            }
            
            # Basic dataset overview
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            results["dataset_overview"] = {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2
            }
            
            # Analyze variable types and characteristics
            variable_analysis = {}
            for col in self.df.columns:
                col_info = {
                    "data_type": str(self.df[col].dtype),
                    "unique_values": self.df[col].nunique(),
                    "missing_count": self.df[col].isnull().sum(),
                    "missing_percentage": (self.df[col].isnull().sum() / len(self.df)) * 100
                }
                
                if col in numeric_cols:
                    col_info.update({
                        "mean": self.df[col].mean(),
                        "std": self.df[col].std(),
                        "min": self.df[col].min(),
                        "max": self.df[col].max(),
                        "skewness": self.df[col].skew()
                    })
                elif col in categorical_cols:
                    col_info.update({
                        "most_frequent": self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else None,
                        "frequency_of_most": self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0
                    })
                
                variable_analysis[col] = col_info
            
            results["variable_types"] = variable_analysis
            
            # Identify key patterns
            patterns = []
            
            # Find potential target variables (binary or categorical with few categories)
            for col in self.df.columns:
                if self.df[col].nunique() == 2:
                    patterns.append(f"Binary variable detected: {col} - potential target for analysis")
                elif self.df[col].nunique() <= 10 and self.df[col].dtype == 'object':
                    patterns.append(f"Low-cardinality categorical: {col} - good for grouping analysis")
            
            # Find highly correlated numeric variables
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols].corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            patterns.append(f"Strong correlation: {corr_matrix.columns[i]} & {corr_matrix.columns[j]} (r={corr_val:.3f})")
            
            # Find variables with high missing data
            for col, info in variable_analysis.items():
                if info["missing_percentage"] > 20:
                    patterns.append(f"High missing data: {col} ({info['missing_percentage']:.1f}% missing)")
            
            results["key_patterns"] = patterns
            
            # Generate focus recommendations based on patterns
            recommendations = []
            if any("Binary variable" in p for p in patterns):
                recommendations.append("Focus on comparing groups using binary variables")
            if any("Strong correlation" in p for p in patterns):
                recommendations.append("Investigate relationships between correlated variables")
            if any("High missing data" in p for p in patterns):
                recommendations.append("Address data quality issues before analysis")
            if len(categorical_cols) > 0:
                recommendations.append("Perform categorical analysis and group comparisons")
            if len(numeric_cols) > 0:
                recommendations.append("Analyze distributions and statistical properties")
                
            results["recommended_focus"] = recommendations
            
            return results
        
        def _visualization_analysis(self) -> Dict:
            """Determine optimal visualizations for the analysis objective."""
            results = {
                "recommended_charts": [],
                "chart_priorities": {},
                "visualization_strategy": ""
            }
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Determine best charts based on data characteristics
            chart_recommendations = []
            
            # For distributions
            for col in numeric_cols:
                if self.df[col].nunique() > 10:  # Continuous
                    chart_recommendations.append({
                        "chart_type": "histogram",
                        "variables": [col],
                        "purpose": f"Show distribution of {col}",
                        "priority": "high" if self.df[col].std() > 0 else "low"
                    })
            
            # For categorical frequencies  
            for col in categorical_cols:
                if self.df[col].nunique() <= 15:  # Manageable categories
                    chart_recommendations.append({
                        "chart_type": "bar_chart",
                        "variables": [col],
                        "purpose": f"Show frequency of {col} categories",
                        "priority": "medium"
                    })
            
            # For relationships
            if len(numeric_cols) >= 2:
                # Correlation heatmap
                chart_recommendations.append({
                    "chart_type": "correlation_heatmap",
                    "variables": numeric_cols[:10],  # Limit to prevent clutter
                    "purpose": "Show correlations between numeric variables",
                    "priority": "high"
                })
                
                # Scatter plots for top correlated pairs
                corr_matrix = self.df[numeric_cols].corr()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > 0.3:  # Moderate correlation
                            chart_recommendations.append({
                                "chart_type": "scatter_plot",
                                "variables": [corr_matrix.columns[i], corr_matrix.columns[j]],
                                "purpose": f"Explore relationship between {corr_matrix.columns[i]} and {corr_matrix.columns[j]}",
                                "priority": "high" if corr_val > 0.7 else "medium"
                            })
            
            # For group comparisons
            for cat_col in categorical_cols:
                if self.df[cat_col].nunique() <= 8:  # Reasonable number of groups
                    for num_col in numeric_cols:
                        chart_recommendations.append({
                            "chart_type": "box_plot",
                            "variables": [cat_col, num_col],
                            "purpose": f"Compare {num_col} across {cat_col} groups",
                            "priority": "medium"
                        })
            
            # Sort by priority
            priority_order = {"high": 3, "medium": 2, "low": 1}
            chart_recommendations.sort(key=lambda x: priority_order[x["priority"]], reverse=True)
            
            results["recommended_charts"] = chart_recommendations[:12]  # Top 12 charts
            
            # Create priority summary
            results["chart_priorities"] = {
                "high_priority": len([c for c in chart_recommendations if c["priority"] == "high"]),
                "medium_priority": len([c for c in chart_recommendations if c["priority"] == "medium"]),
                "low_priority": len([c for c in chart_recommendations if c["priority"] == "low"])
            }
            
            # Create visualization strategy
            strategy_parts = []
            if len(numeric_cols) > 0:
                strategy_parts.append("Start with distribution analysis of key numeric variables")
            if len(categorical_cols) > 0:
                strategy_parts.append("Examine categorical variable frequencies and patterns")
            if len(numeric_cols) >= 2:
                strategy_parts.append("Explore relationships between numeric variables")
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                strategy_parts.append("Compare numeric variables across categorical groups")
            
            results["visualization_strategy"] = ". ".join(strategy_parts) + "."
            
            return results
        
        def generate_ai_interpretation(self, analysis_results: Dict, context: str = "") -> str:
            """Generate AI interpretation of the analysis results."""
            prompt = f"""
            You are a {self.role}.
            
            Your goal: {self.goal}
            Your background: {self.backstory}
            
            Context from previous agents: {context}
            
            I have performed detailed statistical analysis. Here are the ACTUAL ANALYSIS RESULTS:
            
            {str(analysis_results)}
            
            Based on these REAL statistical findings, provide expert interpretation and insights.
            Focus on the actual numbers, statistical significance, and practical implications.
            Be specific about the quantitative findings and what they mean.
            """
            
            try:
                response = self.llm.invoke(prompt)
                result = response.content
                self.memory.append(f"Analysis: {str(analysis_results)}\nInterpretation: {result}")
                return result
            except Exception as e:
                return f"❌ Agent {self.role} interpretation failed: {str(e)}"

    class ObjectiveDrivenAnalysisCrew:
        """Orchestrates objective-focused data analysis agents."""
        
        def __init__(self, agents: List[DataAnalysisAgent], analysis_objective: str):
            self.agents = agents
            self.analysis_objective = analysis_objective
            self.results = []
            self.context = ""
            self.analysis_data = {}
            self.report_sections = {}
        
        def kickoff(self, df: pd.DataFrame) -> List[str]:
            """Execute objective-driven analysis with agents."""
            print(f"\n*** Starting Objective-Driven Analysis: {self.analysis_objective}")
            print("=" * 80)
            
            # Set dataframe for all agents
            for agent in self.agents:
                agent.set_dataframe(df)
            
            for i, agent in enumerate(self.agents, 1):
                print(f"\n{'='*70}")
                print(f"*** Agent {i}: {agent.role}")
                print("*** Performing analysis...")
                
                # Execute analysis
                analysis_results = agent.execute_code_analysis(f"Objective: {self.analysis_objective}")
                
                # Store analysis data
                self.analysis_data[agent.role] = analysis_results
                
                # Generate interpretation
                print("*** Generating insights...")
                interpretation = agent.generate_ai_interpretation(analysis_results, self.context)
                
                self.results.append(interpretation)
                
                # Build context for next agents
                self.context += f"\n\n=== {agent.role} Analysis ===\n"
                self.context += f"Objective: {self.analysis_objective}\n"
                self.context += f"Results: {str(analysis_results)}\n"
                self.context += f"Insights: {interpretation}\n"
                
                print(f"SUCCESS: {agent.role} completed!")
            
            return self.results
        
        def generate_consolidated_report(self, df: pd.DataFrame, file_path: str) -> str:
            """Generate a comprehensive analysis report in Markdown format."""
            report_content = []
            
            # Header
            report_content.append("# 📊 Comprehensive Data Analysis Report")
            report_content.append("")
            report_content.append("---")
            report_content.append("")
            report_content.append("## 📋 Analysis Overview")
            report_content.append("")
            report_content.append(f"**Analysis Objective:** {self.analysis_objective}")
            report_content.append(f"**Dataset:** `{file_path}`")
            report_content.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            
            # Executive Summary
            report_content.append("## 📈 Executive Summary")
            report_content.append("")
            report_content.append(f"- **Dataset Size:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            report_content.append(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            report_content.append(f"- **Data Types:** {len(df.select_dtypes(include=[np.number]).columns)} numeric, {len(df.select_dtypes(include=['object', 'category']).columns)} categorical")
            
            # Add summary from insights agent
            insights_agent = next((agent for agent in self.agents if "Insights" in agent.role), None)
            if insights_agent and insights_agent.analysis_results:
                data_story = insights_agent.analysis_results.get("data_story", "")
                if data_story:
                    report_content.append("")
                    report_content.append(f"**Key Findings:** {data_story}")
            
            report_content.append("")
            
            # Detailed Findings
            for i, (agent, result) in enumerate(zip(self.agents, self.results), 1):
                # Clean up role name for display
                role_name = agent.role.replace("&", "and")
                report_content.append(f"## {i}. 🔍 {role_name}")
                report_content.append("")
                
                # Add quantitative results in a nice format
                if agent.role in self.analysis_data:
                    analysis = self.analysis_data[agent.role]
                    if isinstance(analysis, dict) and not analysis.get("error"):
                        # Create summary stats table or key metrics
                        if "Discovery" in agent.role and "dataset_overview" in analysis:
                            overview = analysis["dataset_overview"]
                            report_content.append("### 📊 Dataset Overview")
                            report_content.append("")
                            report_content.append("| Metric | Value |")
                            report_content.append("|--------|-------|")
                            for key, value in overview.items():
                                key_formatted = key.replace("_", " ").title()
                                if isinstance(value, float):
                                    value_formatted = f"{value:.2f}"
                                elif isinstance(value, int):
                                    value_formatted = f"{value:,}"
                                else:
                                    value_formatted = str(value)
                                report_content.append(f"| {key_formatted} | {value_formatted} |")
                            report_content.append("")
                        
                        elif "Quality" in agent.role and "data_quality" in analysis:
                            quality = analysis["data_quality"]
                            report_content.append("### ✅ Data Quality Assessment")
                            report_content.append("")
                            report_content.append(f"- **Quality Score:** {quality.get('completeness_score', 0):.1%}")
                            report_content.append(f"- **Quality Grade:** {quality.get('quality_grade', 'N/A')}")
                            if quality.get('type_issues'):
                                report_content.append("- **Type Issues:**")
                                for issue in quality['type_issues'][:5]:
                                    report_content.append(f"  - {issue}")
                            report_content.append("")
                        
                        elif "Statistical" in agent.role and "correlation_analysis" in analysis:
                            corr_analysis = analysis["correlation_analysis"]
                            if corr_analysis.get("strong_correlations"):
                                report_content.append("### 🔗 Strong Correlations")
                                report_content.append("")
                                report_content.append("| Variable 1 | Variable 2 | Correlation |")
                                report_content.append("|------------|------------|-------------|")
                                for corr in corr_analysis["strong_correlations"][:5]:
                                    report_content.append(f"| {corr['var1']} | {corr['var2']} | {corr['correlation']:.3f} |")
                                report_content.append("")
                        
                        elif "Visualization" in agent.role and "recommended_charts" in analysis:
                            charts = analysis["recommended_charts"]
                            high_priority_charts = [c for c in charts if c.get("priority") == "high"]
                            if high_priority_charts:
                                report_content.append("### 📈 Recommended Visualizations")
                                report_content.append("")
                                for chart in high_priority_charts[:5]:
                                    report_content.append(f"- **{chart['chart_type'].replace('_', ' ').title()}:** {chart['purpose']}")
                                report_content.append("")
                
                # Add interpretation
                report_content.append("### 💡 Expert Analysis")
                report_content.append("")
                # Format the result text for better readability
                formatted_result = result.replace('\n\n', '\n\n> ').replace('\n', '\n> ')
                report_content.append(f"> {formatted_result}")
                report_content.append("")
            
            # Recommendations
            report_content.append("## 🎯 Actionable Recommendations")
            report_content.append("")
            
            all_recommendations = []
            for agent in self.agents:
                if agent.analysis_results and isinstance(agent.analysis_results, dict):
                    recommendations = agent.analysis_results.get("actionable_recommendations", [])
                    if not recommendations:
                        recommendations = agent.analysis_results.get("recommendations", [])
                    all_recommendations.extend(recommendations)
            
            if all_recommendations:
                for i, rec in enumerate(all_recommendations[:10], 1):
                    report_content.append(f"{i}. **{rec}**")
            else:
                report_content.append("- Continue monitoring data patterns")
                report_content.append("- Implement regular data quality checks")
                report_content.append("- Consider additional data sources for deeper insights")
            
            report_content.append("")
            
            # Conclusion
            report_content.append("## 🎉 Conclusion")
            report_content.append("")
            report_content.append(f"This comprehensive analysis successfully addressed the objective: **{self.analysis_objective}**")
            report_content.append("")
            report_content.append("The findings provide a thorough understanding of the data patterns, quality, and key insights. The statistical analysis reveals important relationships and trends that can inform decision-making.")
            report_content.append("")
            report_content.append("### Next Steps")
            report_content.append("- Review the actionable recommendations above")
            report_content.append("- Implement suggested data quality improvements")
            report_content.append("- Consider the insights for strategic planning")
            report_content.append("- Schedule regular analysis updates as new data becomes available")
            report_content.append("")
            report_content.append("---")
            report_content.append("*Report generated by AI-powered Data Analysis System*")
            
            # Save report as Markdown
            report_text = "\n".join(report_content)
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"data_analysis_report_{timestamp}.md"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            print(f"\n*** MARKDOWN REPORT SAVED: {report_filename}")
            return report_filename

    def main():
        print("*** Welcome to the Objective-Driven Data Analysis System ***")
        print("=" * 60)
        
        # Get user's primary analysis objective
        print("\n*** STEP 1: Define Your Analysis Objective ***")
        print("What is your primary goal for this data analysis?")
        print("\nCommon objectives:")
        print("1. Understand patterns and trends")
        print("2. Compare groups or categories") 
        print("3. Find relationships between variables")
        print("4. Identify outliers and anomalies")
        print("5. Explore data quality and completeness")
        print("6. Business performance analysis")
        print("7. Customer behavior analysis")
        print("8. Custom objective")
        
        objective_choice = input("\nEnter number (1-8) or describe your objective: ").strip()
        
        # Map objectives to analysis focus
        objective_mapping = {
            "1": "pattern_analysis",
            "2": "group_comparison", 
            "3": "relationship_analysis",
            "4": "outlier_detection",
            "5": "data_quality",
            "6": "business_performance",
            "7": "customer_behavior",
            "8": "custom"
        }
        
        if objective_choice in objective_mapping:
            if objective_choice == "8":
                custom_objective = input("Describe your specific objective: ").strip()
                analysis_objective = f"custom: {custom_objective}"
            else:
                objectives = {
                    "1": "Understand patterns and trends in the data",
                    "2": "Compare different groups or categories",
                    "3": "Find relationships between variables", 
                    "4": "Identify outliers and anomalies",
                    "5": "Explore data quality and completeness",
                    "6": "Analyze business performance metrics",
                    "7": "Understand customer behavior patterns"
                }
                analysis_objective = objectives[objective_choice]
        else:
            analysis_objective = f"custom: {objective_choice}"
        
        print(f"\n*** Analysis Objective: {analysis_objective}")
        
        # Check for default dataset (try multiple possible paths)
        print("\n*** STEP 2: Load Dataset ***")
        possible_paths = ["data/titanic.csv", "da/data/titanic.csv", "titanic.csv"]
        default_file = None
        
        for path in possible_paths:
            if os.path.exists(path):
                default_file = path
                break
        
        if default_file:
            print(f"*** Default dataset available: {default_file}")
            choice = input("*** Press Enter to use default dataset, or enter custom file path: ").strip()
            file_path = choice if choice else default_file
        else:
            file_path = input("*** Enter the path to your CSV file: ")

        if not os.path.exists(file_path):
            print("ERROR: File not found.")
            return

        print("SUCCESS: File found. Beginning objective-driven analysis...")
        
        # Set up API
        api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyDloWhFDYK4MGsnAz3MFFUVmUaYlLSeyC8")
        os.environ["GOOGLE_API_KEY"] = api_key
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        
        # Load data
        print("*** Loading dataset for analysis...")
        df = pd.read_csv(file_path)
        print(f"SUCCESS: Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # Create objective-driven analysis agents (no ML)
        agents = [
            DataAnalysisAgent(
                role="Data Discovery & Profiling Analyst",
                goal=f"Understand dataset structure and identify key variables relevant to: {analysis_objective}",
                backstory="Expert in data profiling and automated pattern discovery",
                llm=llm
            ),
            DataAnalysisAgent(
                role="Data Quality & Preparation Analyst",
                goal=f"Assess data quality and prepare clean data for: {analysis_objective}",
                backstory="PhD in Statistics with expertise in data cleaning and quality assessment",
                llm=llm
            ),
            DataAnalysisAgent(
                role="Statistical Analysis Specialist",
                goal=f"Perform comprehensive statistical analysis focused on: {analysis_objective}",
                backstory="Senior statistician specializing in exploratory data analysis and hypothesis testing",
                llm=llm
            ),
            DataAnalysisAgent(
                role="Data Visualization Expert",
                goal=f"Create optimal visualizations that best communicate insights for: {analysis_objective}",
                backstory="Data visualization specialist with expertise in choosing the right charts for different objectives",
                llm=llm
            ),
            DataAnalysisAgent(
                role="Insights & Reporting Analyst",
                goal=f"Synthesize findings and create actionable insights for: {analysis_objective}",
                backstory="Business analyst with expertise in translating data findings into clear, actionable reports",
                llm=llm
            )
        ]
        
        # Create and run the objective-driven crew
        crew = ObjectiveDrivenAnalysisCrew(agents, analysis_objective)
        results = crew.kickoff(df)
        
        # Display comprehensive results
        print(f"\n{'='*90}")
        print("*** OBJECTIVE-DRIVEN ANALYSIS RESULTS ***")
        print(f"{'='*90}")
        
        for i, (agent, result) in enumerate(zip(agents, results), 1):
            print(f"\n*** {i}. {agent.role.upper()}")
            print(f"{'─'*70}")
            
            # Display key findings
            if agent.role in crew.analysis_data:
                analysis = crew.analysis_data[agent.role]
                if isinstance(analysis, dict) and not analysis.get("error"):
                    print("*** KEY FINDINGS:")
                    print(f"{'─'*20}")
                    
                    # Show most relevant info based on role
                    if "Discovery" in agent.role:
                        patterns = analysis.get("key_patterns", [])
                        for pattern in patterns[:5]:
                            print(f"  • {pattern}")
                    elif "Quality" in agent.role:
                        recommendations = analysis.get("recommendations", [])
                        for rec in recommendations:
                            print(f"  • {rec}")
                    elif "Statistical" in agent.role:
                        if "correlation_analysis" in analysis:
                            strong_corrs = analysis["correlation_analysis"].get("strong_correlations", [])
                            for corr in strong_corrs[:3]:
                                print(f"  • Strong correlation: {corr['var1']} & {corr['var2']} (r={corr['correlation']:.3f})")
                    elif "Visualization" in agent.role:
                        charts = analysis.get("recommended_charts", [])
                        high_priority = [c for c in charts if c.get("priority") == "high"]
                        for chart in high_priority[:5]:
                            print(f"  • {chart['chart_type']}: {chart['purpose']}")
            
            print(f"\n*** EXPERT INSIGHTS:")
            print(f"{'─'*25}")
            print(result[:500] + "..." if len(result) > 500 else result)
            print("\n" + "="*70)
        
        # Create objective-driven visualizations
        print("\n*** Generating Objective-Based Visualizations...")
        
        # Get visualization recommendations
        viz_agent = next((agent for agent in agents if "Visualization" in agent.role), None)
        recommended_charts = []
        
        if viz_agent and viz_agent.analysis_results:
            recommended_charts = viz_agent.analysis_results.get("recommended_charts", [])
        
        # Create charts based on recommendations
        if recommended_charts:
            n_charts = min(9, len([c for c in recommended_charts if c.get("priority") in ["high", "medium"]]))
            rows = max(1, int(np.ceil(n_charts / 3)))
            cols = min(3, n_charts)
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
            
            # Handle single subplot case
            if n_charts == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if isinstance(axes, np.ndarray) else [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'Data Analysis Dashboard - {analysis_objective}', fontsize=16, fontweight='bold')
            
            chart_idx = 0
            
            for chart_rec in recommended_charts:
                if chart_idx >= n_charts:
                    break
                    
                if chart_rec.get("priority") not in ["high", "medium"]:
                    continue
                    
                row, col = divmod(chart_idx, 3)
                ax = axes[row, col] if rows > 1 else axes[col]
                
                try:
                    chart_type = chart_rec["chart_type"]
                    variables = chart_rec["variables"]
                    
                    if chart_type == "histogram" and len(variables) == 1:
                        ax.hist(df[variables[0]].dropna(), bins=30, alpha=0.7, edgecolor='black')
                        ax.set_title(f'Distribution of {variables[0]}')
                        ax.set_xlabel(variables[0])
                        ax.set_ylabel('Frequency')
                    
                    elif chart_type == "bar_chart" and len(variables) == 1:
                        value_counts = df[variables[0]].value_counts()
                        value_counts.plot(kind='bar', ax=ax, alpha=0.8)
                        ax.set_title(f'Frequency of {variables[0]}')
                        ax.set_xlabel(variables[0])
                        ax.set_ylabel('Count')
                        ax.tick_params(axis='x', rotation=45)
                    
                    elif chart_type == "correlation_heatmap":
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 1:
                            corr_matrix = df[numeric_cols].corr()
                            sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                                    fmt='.2f', ax=ax, square=True)
                            ax.set_title('Correlation Matrix')
                    
                    elif chart_type == "scatter_plot" and len(variables) == 2:
                        df.plot.scatter(x=variables[0], y=variables[1], ax=ax, alpha=0.6)
                        ax.set_title(f'{variables[0]} vs {variables[1]}')
                    
                    elif chart_type == "box_plot" and len(variables) == 2:
                        df.boxplot(column=variables[1], by=variables[0], ax=ax)
                        ax.set_title(f'{variables[1]} by {variables[0]}')
                        ax.set_xlabel(variables[0])
                        ax.set_ylabel(variables[1])
                    
                    chart_idx += 1
                    
                except Exception as e:
                    print(f"Warning: Could not create {chart_type} chart: {e}")
                    ax.text(0.5, 0.5, f"Chart creation failed:\n{chart_type}", 
                        ha='center', va='center', transform=ax.transAxes)
                    chart_idx += 1
            
            # Hide empty subplots
            total_subplots = rows * 3
            for idx in range(chart_idx, total_subplots):
                row, col = divmod(idx, 3)
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.set_visible(False)
            
            plt.tight_layout()
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            viz_filename = f"objective_analysis_dashboard_{timestamp}.png"
            plt.savefig(viz_filename, dpi=150, bbox_inches='tight')
            plt.show()
            print(f"*** Visualization dashboard saved as: {viz_filename}")
        
        # Generate consolidated report
        print(f"\n{'='*70}")
        print("*** GENERATING COMPREHENSIVE REPORT ***")
        print(f"{'='*70}")
        
        report_filename = crew.generate_consolidated_report(df, file_path)
        
        print(f"\n{'='*70}")
        print("*** ANALYSIS COMPLETE ***")
        print(f"Objective: {analysis_objective}")
        print(f"Markdown Report saved: {report_filename}")
        print(f"Visualizations: Available in PNG files")
        print(f"{'='*70}")
        
        # Simple Q&A (optional)
        print(f"\n*** Quick Q&A Session (optional) ***")
        print("Ask a question about your analysis, or press Enter to exit:")
        
        qa_agent = DataAnalysisAgent(
            role="Data Analysis Consultant",
            goal="Answer questions about the completed analysis",
            backstory="Expert in interpreting data analysis results",
            llm=llm
        )
        qa_agent.set_dataframe(df)
        
        while True:
            question = input("\n*** Your question (or press Enter to exit): ").strip()
            if not question:
                break
                
            # Quick analysis for the question
            try:
                answer = qa_agent.generate_ai_interpretation(
                    crew.analysis_data,
                    f"Analysis objective: {analysis_objective}\nQuestion: {question}\nContext: {crew.context}"
                )
                print(f"\n*** Answer: {answer[:800]}{'...' if len(answer) > 800 else ''}")
            except Exception as e:
                print(f"Sorry, couldn't process that question: {e}")
        
        print("\n*** Thank you for using the Objective-Driven Data Analysis System! ***")

    if __name__ == "__main__":
        main()
