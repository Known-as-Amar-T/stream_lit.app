import os
from io import StringIO, BytesIO
from streamlit_agraph import agraph, Node, Edge, Config
import PyPDF2
import docx
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import openpyxl

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(model, documents):
    return np.array([model.encode(doc["content"]) for doc in documents])

def find_similar_incidents(query, model, documents, embeddings, top_k=3):
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [(documents[i], similarities[i]) for i in top_indices]

def extract_data_from_content(content):
    lines = content.split('\n')
    data = {
        'title': lines[0] if len(lines) > 0 else '',
        'incident_start_time': lines[1] if len(lines) > 1 else '',
        'incident_detected_time': lines[2] if len(lines) > 2 else '',
        'incident_end_time': lines[3] if len(lines) > 3 else '',
        'severity_level': lines[4] if len(lines) > 4 else '',
        'summary': lines[5] if len(lines) > 5 else '',
        'incident_duration': lines[6] if len(lines) > 6 else '',
        'time_to_detect': lines[7] if len(lines) > 7 else '',
        'time_to_resolve': lines[8] if len(lines) > 8 else '',
        'services': lines[9] if len(lines) > 9 else '',
        'services_impact_type': lines[10] if len(lines) > 10 else '',
        'components': lines[11] if len(lines) > 11 else '',
        'components_root_cause': lines[12] if len(lines) > 12 else '',
        'impact': lines[13] if len(lines) > 13 else '',
        'user_impact_type': lines[14] if len(lines) > 14 else '',
        'root_cause': lines[15] if len(lines) > 15 else '',
        'category_root_cause': lines[16] if len(lines) > 16 else '',
        'symptom': lines[17] if len(lines) > 17 else '',
        'symptom_type': lines[18] if len(lines) > 18 else ''
    }
    return data

def analyze_data(documents):
    data = []
    for doc in documents:
        content = doc['content']
        extracted_data = extract_data_from_content(content)
        extracted_data['filename'] = doc['filename']
        data.append(extracted_data)
    return data

def file_uploader_page():
    st.title("Bulk Text File Analyzer")
    uploaded_files = st.file_uploader("Choose .txt files", type="txt", accept_multiple_files=True)

    if uploaded_files:
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = []

        for uploaded_file in uploaded_files:
            string_data = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            st.session_state.uploaded_data.append({'filename': uploaded_file.name, 'content': string_data})

        st.write(f"Number of files uploaded: {len(st.session_state.uploaded_data)}")

        # Analyze the uploaded data
        analyzed_data = analyze_data(st.session_state.uploaded_data)

        # Create a DataFrame from the analyzed data
        df = pd.DataFrame(analyzed_data)

        # Save the DataFrame to an Excel file in memory
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')

            # Apply formatting
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            format1 = workbook.add_format({'text_wrap': True, 'valign': 'top'})
            worksheet.set_column('A:Z', 20, format1)

        st.session_state.excel_file = excel_buffer.getvalue()

        st.write("Excel file created from uploaded .txt files.")

def dashboard_page():
    st.title("Incidents Insights -- Dashboard")
    st.subheader("Incident Stats Last 12 months")
    st.write("Count: 10")
    st.write("MTTR: 5.45 hours")
    st.write("MTTD: 3.25 hours")

    st.subheader("Incident Analysis Over Time")
    st.write("Frequency of Incidents")

    data = {
        'Year': ['2020'] * 12 + ['2021'] * 12,
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] * 2,
        'Incidents': [5, 7, 8, 6, 9, 4, 3, 5, 6, 7, 8, 9, 6, 8, 7, 5, 10, 6, 4, 7, 8, 9, 10, 11]
    }
    df = pd.DataFrame(data)
    df = df.sort_values(by='Incidents', ascending=False)
    fig = px.bar(df, x='Year', y='Incidents', color='Month', barmode='group', title='Monthly Incident Count by Year', labels={'Incidents': 'Number of Incidents'})
    fig.update_traces(hovertemplate='Year: %{x}<br>Month: %{marker.color}<br>Incidents: %{y}')
    st.plotly_chart(fig)

    st.header("What Components Fail the Most?")
    st.subheader("Incident Distribution by Component Root Cause")

    component_data = {
        'Component': ['Networking', 'Networking', 'MediaWiki', 'MediaWiki', 'Database Systems', 'Database Systems', 'Web Servers', 'Web Servers', 'CDN & Caching', 'CDN & Caching', 'Storage', 'Storage'],
        'Failures': [10, 8, 15, 10, 12, 8, 20, 15, 12, 8, 15, 10],
        'Root Cause': ['Network Issues', 'Configuration Issues', 'Software Bugs', 'Performance Issues', 'Database Corruption', 'Connection Issues', 'Server Overload', 'Configuration Issues', 'Cache Issues', 'Performance Issues', 'Storage Failures', 'Hardware Issues']
    }
    component_df = pd.DataFrame(component_data)
    component_df_grouped = component_df.groupby(['Component', 'Root Cause'])['Failures'].sum().reset_index().sort_values(by='Failures', ascending=False)
    component_fig = px.bar(component_df_grouped, x='Component', y='Failures', color='Root Cause', title='Component Failures by Root Cause', labels={'Failures': 'Number of Failures'}, barmode='stack')
    component_fig.update_traces(hovertemplate="<b>%{x}</b><br>Root Cause: %{marker.color}<br>Failures: %{y}<br><extra></extra>")
    st.plotly_chart(component_fig)
    st.subheader("Detailed Breakdown by Component")
    st.dataframe(component_df_grouped.sort_values(['Failures'], ascending=False), hide_index=True)

    st.subheader("What Symptoms are most Common?")
    st.write("Incident Distribution by Symptom Type")

    symptom_data = {
        'Symptom': ['Service Unavailable', 'Service Unavailable', 'Service Unavailable', 'Slow Performance', 'Slow Performance', 'Slow Performance', 'Crash', 'Crash', 'Crash', 'Data Loss', 'Data Loss', 'Data Loss', 'Security Breach', 'Security Breach', 'Security Breach', 'Network Outage', 'Network Outage', 'Network Outage', 'High CPU Usage', 'High CPU Usage', 'High CPU Usage', 'Memory Leak', 'Memory Leak', 'Memory Leak'],
        'Occurrences': [10, 12, 8, 20, 18, 17, 15, 10, 5, 8, 7, 5, 6, 5, 9, 12, 10, 13, 8, 7, 15, 5, 4, 11],
        'Category': ['Availability', 'Performance', 'Network', 'Performance', 'Availability', 'Network', 'Stability', 'Performance', 'Network', 'Data', 'Performance', 'Network', 'Security', 'Performance', 'Network', 'Network', 'Performance', 'Availability', 'Resource', 'Performance', 'Network', 'Memory', 'Performance', 'Network']
    }
    symptom_df = pd.DataFrame(symptom_data).sort_values(by='Occurrences', ascending=False)
    symptom_fig = px.bar(symptom_df, x='Symptom', y='Occurrences', color='Category', title='Incident Symptoms Distribution', labels={'Occurrences': 'Number of Incidents'}, barmode='stack')
    st.plotly_chart(symptom_fig)
    st.subheader("Detailed Breakdown by Symptom")
    st.dataframe(symptom_df.sort_values(['Occurrences'], ascending=False), hide_index=True)

    st.header("What Services and Factors are Affecting Incidents?")
    st.subheader("Service Impact Analysis")

    service_factors_data = {
        'Service': ['Web Services', 'Web Services', 'Web Services', 'Database', 'Database', 'Database', 'Storage', 'Storage', 'Storage', 'Network', 'Network', 'Network'],
        'Factor': ['Resource Issues', 'Configuration', 'Performance', 'Capacity', 'Performance', 'Connectivity', 'Disk Space', 'IO Issues', 'Hardware', 'Bandwidth', 'Hardware', 'Configuration'],
        'Impact': [20, 15, 25, 30, 20, 15, 25, 20, 15, 30, 20, 15],
        'Severity': ['High', 'Medium', 'Critical', 'Critical', 'High', 'Medium', 'High', 'Medium', 'Critical', 'Critical', 'High', 'Medium']
    }
    service_df = pd.DataFrame(service_factors_data).sort_values(by='Impact', ascending=False)
    service_fig = px.bar(service_df, x='Service', y='Impact', color='Factor', pattern_shape='Severity', title='Service Impact Analysis', labels={'Impact': 'Impact Score'}, barmode='stack')
    service_fig.update_layout(xaxis_title="Services", yaxis_title="Impact Score", showlegend=True, legend_title="Factors")
    st.plotly_chart(service_fig)
    st.subheader("Detailed Service Impact Breakdown")
    st.dataframe(service_df.sort_values(['Impact'], ascending=False), hide_index=True)

    st.subheader("Where were Users Impacted the Most?")
    st.write("Incidents Distribution by User Impact")

    service_issues_data = {
        'Issue Type': ['Specific Failures', 'Specific Failures', 'Specific Failures', 'Performance Degradation', 'Performance Degradation', 'Performance Degradation', 'Outage', 'Outage', 'Outage', 'Network Issues', 'Network Issues', 'Network Issues', 'API Issues', 'API Issues', 'API Issues', 'Caching Issues', 'Caching Issues', 'Caching Issues', 'Jobs and Queues', 'Jobs and Queues', 'Jobs and Queues', 'Database Issues', 'Database Issues', 'Database Issues', 'Data Loss/Corruption', 'Data Loss/Corruption', 'Data Loss/Corruption', 'Deployment Issues', 'Deployment Issues', 'Deployment Issues', 'Authentication Issues', 'Authentication Issues', 'Authentication Issues', 'Monitoring and Observability', 'Monitoring and Observability', 'Monitoring and Observability', 'Security Incidents', 'Security Incidents', 'Security Incidents', 'Others', 'Others', 'Others'],
        'Incidents': [5, 4, 3, 10, 8, 7, 6, 5, 7, 8, 6, 6, 4, 5, 6, 3, 4, 3, 2, 3, 3, 7, 5, 6, 4, 5, 5, 3, 4, 2, 2, 3, 2, 1, 2, 1, 3, 2, 2, 1, 1, 1],
        'Impact': ['Resource Issues', 'Performance', 'Availability', 'Performance', 'Availability', 'Network', 'Availability', 'Network', 'API', 'Network', 'API', 'Caching', 'API', 'Caching', 'Jobs', 'Caching', 'Jobs', 'Database', 'Jobs', 'Database', 'Data Loss', 'Database', 'Data Loss', 'Deployment', 'Data Loss', 'Deployment', 'Authentication', 'Deployment', 'Authentication', 'Monitoring', 'Authentication', 'Monitoring', 'Security', 'Monitoring', 'Security', 'Other', 'Security', 'Other', 'Resource Issues', 'Other', 'Resource Issues', 'Performance']
    }
    service_issues_df = pd.DataFrame(service_issues_data).sort_values(by='Incidents', ascending=False)
    service_issues_fig = px.bar(service_issues_df, x='Issue Type', y='Incidents', color='Impact', title='Where were Users Impacted the Most?', labels={'Incidents': 'Number of Incidents'}, barmode='stack')
    service_issues_fig.update_traces(hovertemplate="<b>%{x}</b><br>Incidents: %{y}<br>Impact: %{marker.color}<br><extra></extra>")
    st.plotly_chart(service_issues_fig)
    st.subheader("Detailed Breakdown by Issue Type")
    st.dataframe(service_issues_df.sort_values(['Incidents'], ascending=False), hide_index=True)

    st.subheader("What are the Most Common Root Causes?")
    st.write("Incident Distribution by Category Root Cause")

    root_cause_data = {
        'Root Cause': ['Configuration Error', 'Configuration Error', 'Traffic and Load Surge', 'Traffic and Load Surge', 'Network Problems', 'Network Problems', 'Software Bugs', 'Software Bugs', 'Maintenance and Operational Errors', 'Maintenance and Operational Errors', 'Resource Management', 'Resource Management', 'Database Issues', 'Database Issues', 'Services Dependency Failures', 'Services Dependency Failures', 'API and Misconfiguration', 'API and Misconfiguration', 'Cache Management', 'Cache Management', 'Code Deployment Issues', 'Code Deployment Issues', 'Hardware Failures', 'Hardware Failures', 'Storage Issues', 'Storage Issues', 'Security Error', 'Security Error', 'Monitoring and Logging Failures', 'Monitoring and Logging Failures'],
        'Factor': ['Human Error', 'System Misconfiguration', 'High Traffic', 'Unexpected Load', 'ISP Issues', 'Internal Network', 'Code Bugs', 'Third-Party Software', 'Operational Mistakes', 'Maintenance Errors', 'Resource Allocation', 'Resource Exhaustion', 'Data Corruption', 'Connection Issues', 'Service Failures', 'Dependency Issues', 'API Errors', 'Misconfiguration', 'Cache Misses', 'Cache Overload', 'Deployment Failures', 'Code Issues', 'Hardware Malfunctions', 'Hardware Wear', 'Storage Failures', 'Disk Issues', 'Security Breaches', 'Vulnerabilities', 'Monitoring Failures', 'Logging Issues'],
        'Incidents': [10, 15, 8, 10, 12, 8, 10, 12, 7, 8, 5, 5, 6, 6, 4, 4, 7, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 3, 3]
    }
    root_cause_df = pd.DataFrame(root_cause_data).sort_values(by='Incidents', ascending=False)
    root_cause_fig = px.bar(root_cause_df, x='Root Cause', y='Incidents', color='Factor', title='What are the Most Common Root Causes?', labels={'Incidents': 'Number of Incidents'}, barmode='stack')
    root_cause_fig.update_traces(hovertemplate="<b>%{x}</b><br>Factor: %{marker.color}<br>Incidents: %{y}<br><extra></extra>")
    st.plotly_chart(root_cause_fig)
    st.subheader("Detailed Breakdown by Root Cause")
    st.dataframe(root_cause_df.sort_values(['Incidents'], ascending=False), hide_index=True)

def correlation_page():
    st.title("Interactive Service and Component Correlation Graph")
    st.write("Select a service/component to highlight (Select 'None' to see the whole world)")

    # Define all services and components with common incidents
    services = {
        'Web Services': 15,
        'Database': 20,
        'Storage': 10,
        'Network': 12
    }
    all_components = {
        'Web Services': [('Load Balancer', 5), ('Apache', 3), ('Nginx', 7)],
        'Database': [('MySQL', 8), ('PostgreSQL', 6), ('Redis', 4)],
        'Storage': [('S3', 7), ('EBS', 5), ('File System', 3)],
        'Network': [('DNS', 6), ('CDN', 4), ('Firewall', 2)]
    }

    # Extract unique service/component names
    service_component_names = ["None"] + list(services.keys()) + [comp for comps in all_components.values() for comp, _ in comps]

    # Dropdown for selecting service/component
    selected_service_component = st.selectbox("Select Service/Component", service_component_names)

    related_nodes = []  # Initialize related_nodes to an empty list

    if selected_service_component != "None":
        st.subheader(f"Top 10 Related Nodes for {selected_service_component}")

        # Display related nodes and their common incidents
        st.write("Related Nodes and their Common Incidents:")
        if selected_service_component == 'Web Services':
            related_nodes = all_components['Web Services']
        elif selected_service_component == 'Database':
            related_nodes = all_components['Database']
        elif selected_service_component == 'Storage':
            related_nodes = all_components['Storage']
        elif selected_service_component == 'Network':
            related_nodes = all_components['Network']
        
        for node, incidents in related_nodes:
            st.write(f"{node} :- {incidents} common incidents")

        # Generate nodes and edges for the graph
        nodes = [Node(id=selected_service_component, label=selected_service_component)]
        edges = []
        for node, incidents in related_nodes:
            nodes.append(Node(id=node, label=f"{node} ({incidents} common incidents)"))
            edges.append(Edge(source=selected_service_component, target=node))

        config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.subheader("Top 10 Related Nodes for All Services/Components")

        # Display related nodes and their common incidents
        st.write("Related Nodes and their Common Incidents:")
        for service, comps in all_components.items():
            for comp, count in comps:
                st.write(f"{comp} :- {count} common incidents")

        # Generate nodes and edges for the graph
        nodes = [Node(id=service, label=f"{service} ({count} incidents)") for service, count in services.items()]
        edges = []
        for service, comps in all_components.items():
            for comp, count in comps:
                nodes.append(Node(id=comp, label=f"{comp} ({count} common incidents)"))
                edges.append(Edge(source=service, target=comp))

        config = Config(width=800, height=600, directed=True, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
        agraph(nodes=nodes, edges=edges, config=config)

    # Add interaction hints
    st.write("""
    **Graph Interaction Tips:**
    - Drag nodes to reposition them
    - Zoom with mouse wheel
    - Click on nodes to highlight connections
    - Drag the background to pan
    """)

def related_incidents_page():
    st.title("Find Past Incidents related to your Alert")
    st.write("Chatbot about the documents")

    if 'uploaded_data' not in st.session_state or not st.session_state.uploaded_data:
        st.write("No documents uploaded. Please upload documents in the 'Data Uploader' section.")
        return

    query = st.text_input("Ask me anything about the past incidents in wikiMedia:")
    if query:
        st.write("You asked:", query)
        model = load_model()
        documents = st.session_state.uploaded_data
        embeddings = create_embeddings(model, documents)
        similar_incidents = find_similar_incidents(query, model, documents, embeddings)
        for incident, similarity in similar_incidents:
            st.write(f"Incident: {incident['filename']}, Similarity: {similarity:.2f}")
            st.write(f"Content: {incident['content']}")
            st.write("---")

def raw_data_page():
    st.title("Incident Report Overview")
    st.write("This page will display raw data related to incident reports.")

    if 'excel_file' in st.session_state:
        df = pd.read_excel(BytesIO(st.session_state.excel_file))
        st.dataframe(df)
    else:
        st.write("No data available. Please upload .txt files in the 'Data Uploader' section.")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Uploader", "Dashboard", "Risk Evaluation", "Related Incidents", "Raw Data"])

    if page == "Data Uploader":
        file_uploader_page()
    elif page == "Dashboard":
        dashboard_page()
    elif page == "Risk Evaluation":
        correlation_page()
    elif page == "Related Incidents":
        related_incidents_page()
    elif page == "Raw Data":
        raw_data_page()

if __name__ == "__main__":
    main()