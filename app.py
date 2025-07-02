
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import shap
import plotly.graph_objects as go

st.set_page_config(page_title="YoY Decomposition App", layout="wide")

st.title("📊 YoY Order & Revenue Decomposition App")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file with data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=["Month"])

    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.sidebar.header("Configuration")
    country_col = st.sidebar.selectbox("Select Country Column", df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Columns (Traffic, Promo, etc.)",
                                           [col for col in df.columns if col not in ['Month', country_col, 'Orders', 'Revenue']])
    target = st.sidebar.selectbox("Target Variable", ['Orders', 'Revenue'])

    if feature_cols:
        st.header("Overall YoY Decomposition")

        agg = df.groupby('Month')[[*feature_cols, target]].sum().reset_index()
        agg['Year'] = agg['Month'].dt.year

        years = sorted(agg['Year'].unique())
        if len(years) >= 2:
            last_year = years[-1]
            prev_year = years[-2]

            X = agg[feature_cols]
            y = agg[target]

            model = GradientBoostingRegressor(random_state=42)
            model.fit(X, y)

            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            shap_summary = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': mean_shap})

            avg_features_prev = agg[agg['Year'] == prev_year][feature_cols].mean()
            avg_features_last = agg[agg['Year'] == last_year][feature_cols].mean()
            yoy_feature_change = avg_features_last - avg_features_prev

            contribution = shap_summary.set_index('Feature')['Mean_SHAP'] * yoy_feature_change

            target_change = y[agg['Year'] == last_year].mean() - y[agg['Year'] == prev_year].mean()

            contrib_df = pd.DataFrame({
                'Feature': contribution.index,
                'Contribution': contribution.values
            }).sort_values(by='Contribution', key=abs, ascending=False)

            residual = target_change - contrib_df['Contribution'].sum()
            contrib_df = pd.concat([contrib_df, pd.DataFrame({'Feature': ['Residual'], 'Contribution': [residual]})])

            fig = go.Figure(go.Waterfall(
                name="Feature Attribution",
                orientation="v",
                measure=["relative"] * (len(contrib_df) - 1) + ["total"],
                x=contrib_df['Feature'],
                text=[f"{c:.0f}" for c in contrib_df['Contribution']],
                y=contrib_df['Contribution'],
                connector={"line":{"color":"rgb(63, 63, 63)"}}
            ))

            fig.update_layout(
                title=f"YoY {target} Change Attribution (Total Change: {target_change:.0f})",
                waterfallgap=0.3
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Decomposition Table")
            st.dataframe(contrib_df)

            st.header("Country-Level YoY Decomposition")
            countries = df[country_col].unique()
            country_tabs = st.tabs([str(c) for c in countries])

            for tab, tab_name in zip(country_tabs, countries):
                with tab:
                    country_df = df[df[country_col] == tab_name]

                    if country_df.empty:
                        st.warning(f"No data for {tab_name}.")
                    else:
                        agg_ct = country_df.groupby('Month')[[*feature_cols, target]].sum().reset_index()
                        agg_ct['Year'] = agg_ct['Month'].dt.year

                        years_ct = sorted(agg_ct['Year'].unique())
                        if len(years_ct) >= 2:
                            last_year_ct = years_ct[-1]
                            prev_year_ct = years_ct[-2]

                            X_ct = agg_ct[feature_cols]
                            y_ct = agg_ct[target]

                            model_ct = GradientBoostingRegressor(random_state=42)
                            model_ct.fit(X_ct, y_ct)

                            explainer_ct = shap.Explainer(model_ct, X_ct)
                            shap_values_ct = explainer_ct(X_ct)
                            mean_shap_ct = np.abs(shap_values_ct.values).mean(axis=0)
                            shap_summary_ct = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': mean_shap_ct})

                            avg_features_prev_ct = agg_ct[agg_ct['Year'] == prev_year_ct][feature_cols].mean()
                            avg_features_last_ct = agg_ct[agg_ct['Year'] == last_year_ct][feature_cols].mean()
                            yoy_feature_change_ct = avg_features_last_ct - avg_features_prev_ct

                            contribution_ct = shap_summary_ct.set_index('Feature')['Mean_SHAP'] * yoy_feature_change_ct

                            target_change_ct = y_ct[agg_ct['Year'] == last_year_ct].mean() - y_ct[agg_ct['Year'] == prev_year_ct].mean()

                            contrib_df_ct = pd.DataFrame({
                                'Feature': contribution_ct.index,
                                'Contribution': contribution_ct.values
                            }).sort_values(by='Contribution', key=abs, ascending=False)

                            residual_ct = target_change_ct - contrib_df_ct['Contribution'].sum()
                            contrib_df_ct = pd.concat([contrib_df_ct, pd.DataFrame({'Feature': ['Residual'], 'Contribution': [residual_ct]})])

                            fig_ct = go.Figure(go.Waterfall(
                                name="Feature Attribution",
                                orientation="v",
                                measure=["relative"] * (len(contrib_df_ct) - 1) + ["total"],
                                x=contrib_df_ct['Feature'],
                                text=[f"{c:.0f}" for c in contrib_df_ct['Contribution']],
                                y=contrib_df_ct['Contribution'],
                                connector={"line":{"color":"rgb(63, 63, 63)"}}
                            ))

                            fig_ct.update_layout(
                                title=f"{tab_name} YoY {target} Change Attribution (Total Change: {target_change_ct:.0f})",
                                waterfallgap=0.3
                            )

                            st.plotly_chart(fig_ct, use_container_width=True)
                        else:
                            st.warning("Not enough data for YoY decomposition.")
        else:
            st.warning("Not enough data for YoY decomposition.")
else:
    st.info("Upload a CSV file to get started.")
