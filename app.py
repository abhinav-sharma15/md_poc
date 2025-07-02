import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import shap
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Marketing Impact Decomposition", layout="wide")

st.title("📊 Marketing Channel Impact & YoY Order/Revenue Decomposition")

# Step 1: Data Upload and Initial Inspection
uploaded_file = st.file_uploader("Upload your marketing & order data CSV", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded!")
    st.write("Sample data:", df.head())

    # Required columns check
    required_cols = ['Month', 'Country', 'Paid_Search_Traffic', 'Organic_Traffic', 'Display_Traffic',
                     'Email_Traffic', 'Affiliate_Traffic', 'Promo_Level', 'Offer_Type', 'Orders', 'Revenue']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
        st.stop()

    # Convert Month to datetime
    df['Month'] = pd.to_datetime(df['Month'])

    # Step 2: Filter countries
    countries = df['Country'].unique()
    selected_countries = st.multiselect("Select countries to analyze", options=countries, default=list(countries))

    if selected_countries:
        filtered_df = df[df['Country'].isin(selected_countries)].copy()

        # Step 3: Target selection
        target = st.selectbox("Choose target metric", options=['Orders', 'Revenue'], index=0)

        # Step 4: Prepare features and encode Offer_Type
        features = [
            "Paid_Search_Traffic", "Organic_Traffic", "Display_Traffic",
            "Email_Traffic", "Affiliate_Traffic", "Promo_Level"
        ]

        df_model = pd.get_dummies(filtered_df, columns=["Offer_Type"], drop_first=True)
        feature_cols = features + [col for col in df_model.columns if col.startswith("Offer_Type_")]

        # Step 5: Decomposition Engine
        st.header("📉 Feature Decomposition & Attribution")

        X = df_model[feature_cols]
        y = df_model[target]

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X, y)

        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        mean_shap = np.abs(shap_values.values).mean(axis=0)
        shap_summary = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': mean_shap})

        df_model['Year'] = df_model['Month'].dt.year
        years = sorted(df_model['Year'].unique())

        if len(years) < 2:
            st.warning("Need at least two years of data for YoY decomposition.")
        else:
            last_year = years[-1]
            prev_year = years[-2]

            avg_features_prev = df_model[df_model['Year'] == prev_year][feature_cols].mean()
            avg_features_last = df_model[df_model['Year'] == last_year][feature_cols].mean()
            yoy_feature_change = avg_features_last - avg_features_prev

            contribution = shap_summary.set_index('Feature')['Mean_SHAP'] * yoy_feature_change

            target_change = y[df_model['Year'] == last_year].mean() - y[df_model['Year'] == prev_year].mean()

            contrib_df = pd.DataFrame({
                'Feature': contribution.index,
                'Contribution': contribution.values
            }).sort_values(by='Contribution', key=abs, ascending=False)

            residual = target_change - contrib_df['Contribution'].sum()
            contrib_df = contrib_df.append({'Feature': 'Residual', 'Contribution': residual}, ignore_index=True)

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

        # Step 6: Counterfactual Simulator
        st.header("⚙️ Counterfactual Simulator")

        latest_month = filtered_df['Month'].max()
        baseline_df = filtered_df[filtered_df['Month'] == latest_month].copy()

        for country in selected_countries:
            st.subheader(f"Simulate Inputs for {country}")

            base_row = baseline_df[baseline_df['Country'] == country]
            if base_row.empty:
                st.warning(f"No data for {country} in latest month.")
                continue

            default_values = base_row.iloc[0]

            paid_search = st.slider(f"Paid Search Traffic ({country})", 0, int(default_values.Paid_Search_Traffic * 2), int(default_values.Paid_Search_Traffic))
            organic = st.slider(f"Organic Traffic ({country})", 0, int(default_values.Organic_Traffic * 2), int(default_values.Organic_Traffic))
            display = st.slider(f"Display Traffic ({country})", 0, int(default_values.Display_Traffic * 2), int(default_values.Display_Traffic))
            email = st.slider(f"Email Traffic ({country})", 0, int(default_values.Email_Traffic * 2), int(default_values.Email_Traffic))
            affiliate = st.slider(f"Affiliate Traffic ({country})", 0, int(default_values.Affiliate_Traffic * 2), int(default_values.Affiliate_Traffic))
            promo_level = st.slider(f"Promo Level ({country})", 0, 50, int(default_values.Promo_Level))

            offer_types = filtered_df['Offer_Type'].unique()
            offer = st.selectbox(f"Offer Type ({country})", options=offer_types, index=list(offer_types).tolist().index(default_values.Offer_Type))

            input_dict = {
                "Paid_Search_Traffic": paid_search,
                "Organic_Traffic": organic,
                "Display_Traffic": display,
                "Email_Traffic": email,
                "Affiliate_Traffic": affiliate,
                "Promo_Level": promo_level,
            }

            for ot in [col for col in df_model.columns if col.startswith("Offer_Type_")]:
                input_dict[ot] = 0
            offer_col = f"Offer_Type_{offer}"
            if offer_col in input_dict:
                input_dict[offer_col] = 1

            input_df = pd.DataFrame([input_dict])

            predicted = model.predict(input_df)[0]

            baseline_value = default_values[target]
            pct_change = (predicted - baseline_value) / baseline_value * 100 if baseline_value != 0 else 0

            st.markdown(f"**Predicted {target}:** {predicted:.0f} ({pct_change:+.2f} % change from baseline)")

        # Step 7: Multi-Country Dashboard
        st.header("🌍 Multi-Country Dashboard")

        tabs = st.tabs(selected_countries + ["Combined"])

        for tab_name in tabs:
            with tab_name:
                if tab_name == "Combined":
                    st.subheader("Combined Country View")
                    combined_df = filtered_df.copy()
                    agg_cols = feature_cols + [target]
                    combined_agg = combined_df.groupby('Month')[agg_cols].sum().reset_index()
                    combined_agg['Year'] = combined_agg['Month'].dt.year

                    years_combined = sorted(combined_agg['Year'].unique())
                    if len(years_combined) >= 2:
                        last_year_c = years_combined[-1]
                        prev_year_c = years_combined[-2]

                        X_combined = combined_agg[feature_cols]
                        y_combined = combined_agg[target]

                        model_combined = GradientBoostingRegressor(random_state=42)
                        model_combined.fit(X_combined, y_combined)

                        explainer_combined = shap.Explainer(model_combined, X_combined)
                        shap_values_combined = explainer_combined(X_combined)
                        mean_shap_combined = np.abs(shap_values_combined.values).mean(axis=0)
                        shap_summary_combined = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': mean_shap_combined})

                        avg_features_prev_c = combined_agg[combined_agg['Year'] == prev_year_c][feature_cols].mean()
                        avg_features_last_c = combined_agg[combined_agg['Year'] == last_year_c][feature_cols].mean()
                        yoy_feature_change_c = avg_features_last_c - avg_features_prev_c

                        contribution_c = shap_summary_combined.set_index('Feature')['Mean_SHAP'] * yoy_feature_change_c

                        target_change_c = y_combined[combined_agg['Year'] == last_year_c].mean() - y_combined[combined_agg['Year'] == prev_year_c].mean()

                        contrib_df_c = pd.DataFrame({
                            'Feature': contribution_c.index,
                            'Contribution': contribution_c.values
                        }).sort_values(by='Contribution', key=abs, ascending=False)

                        residual_c = target_change_c - contrib_df_c['Contribution'].sum()
                        contrib_df_c = contrib_df_c.append({'Feature': 'Residual', 'Contribution': residual_c}, ignore_index=True)

                        fig_c = go.Figure(go.Waterfall(
                            name="Feature Attribution",
                            orientation="v",
                            measure=["relative"] * (len(contrib_df_c) - 1) + ["total"],
                            x=contrib_df_c['Feature'],
                            text=[f"{c:.0f}" for c in contrib_df_c['Contribution']],
                            y=contrib_df_c['Contribution'],
                            connector={"line":{"color":"rgb(63, 63, 63)"}}
                        ))

                        fig_c.update_layout(
                            title=f"Combined YoY {target} Change Attribution (Total Change: {target_change_c:.0f})",
                            waterfallgap=0.3
                        )
                        st.plotly_chart(fig_c, use_container_width=True)
                    else:
                        st.warning("Not enough data for combined YoY decomposition.")

                else:
                    st.subheader(f"Country: {tab_name}")
                    country_df = filtered_df[filtered_df['Country'] == tab_name]
                    if country_df.empty:
                        st.warning(f"No data for {tab_name}."}
                    else:
                        agg_cols = feature_cols + [target]
                        agg = country_df.groupby('Month')[agg_cols].sum().reset_index()
                        agg['Year'] = agg['Month'].dt.year

                        years_country = sorted(agg['Year'].unique())
                        if len(years_country) >= 2:
                            last_year_ct = years_country[-1]
                            prev_year_ct = years_country[-2]

                            X_ct = agg[feature_cols]
                            y_ct = agg[target]

                            model_ct = GradientBoostingRegressor(random_state=42)
                            model_ct.fit(X_ct, y_ct)

                            explainer_ct = shap.Explainer(model_ct, X_ct)
                            shap_values_ct = explainer_ct(X_ct)
                            mean_shap_ct = np.abs(shap_values_ct.values).mean(axis=0)
                            shap_summary_ct = pd.DataFrame({'Feature': feature_cols, 'Mean_SHAP': mean_shap_ct})

                            avg_features_prev_ct = agg[agg['Year'] == prev_year_ct][feature_cols].mean()
                            avg_features_last_ct = agg[agg['Year'] == last_year_ct][feature_cols].mean()
                            yoy_feature_change_ct = avg_features_last_ct - avg_features_prev_ct

                            contribution_ct = shap_summary_ct.set_index('Feature')['Mean_SHAP'] * yoy_feature_change_ct

                            target_change_ct = y_ct[agg['Year'] == last_year_ct].mean() - y_ct[agg['Year'] == prev_year_ct].mean()

                            contrib_df_ct = pd.DataFrame({
                                'Feature': contribution_ct.index,
                                'Contribution': contribution_ct.values
                            }).sort_values(by='Contribution', key=abs, ascending=False)

                            residual_ct = target_change_ct - contrib_df_ct['Contribution'].sum()
                            contrib_df_ct = contrib_df_ct.append({'Feature': 'Residual', 'Contribution': residual_ct}, ignore_index=True)

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

        st.header("💾 Export Data & Charts")

        if 'contrib_df' in locals():
            csv = contrib_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Contribution Data",
                csv,
                f"{target}_contribution.csv",
                "text/csv"
            )
        else:
            st.info("Run decomposition to enable download.")
