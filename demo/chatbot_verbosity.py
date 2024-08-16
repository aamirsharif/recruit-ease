import sys
import json
import streamlit as st
import numpy as np

sys.dont_write_bytecode = True

def render(document_list: list, meta_data: dict, time_elapsed):
    retriever_message = st.expander(f"Verbosity")
    message_map = {
        "retrieve_applicant_jd": "**A job description is detected**. The system defaults to using RAG...",
        "retrieve_applicant_id": "**Applicant IDs are provided**. The system defaults to using exact ID retrieval...",
        "no_retrieve": "**No retrieval is required for this task**. The system will utilize chat history to answer..."
    }

    requirements = {
        "Location": [
            "1. Candidate's number should start with 07 or +44.",
            "2. Candidate Must be a resident of England."
        ],
        "Qualification": [
            "1. Candidates must pass the GCSE exam in these countries: UK, IRE, AUS, NZ, CAN, SA.",
            "2. A tertiary qualification."
        ],
        "Experience": [
            "1. Classroom experience or formal Teacher training in one of the following countries within the last 2 years: UK, IRE, AUS, NZ, CAN, SA.",
            "2. Candidate should be a Primary Teacher, Secondary teacher, Teaching assistant, SEN teacher, SEN teaching assistant, LSA, HLTA."
        ]
    }

    with retriever_message:
        st.markdown(f"Total time elapsed: {np.round(time_elapsed, 3)} seconds")
        st.markdown(f"{message_map[meta_data['query_type']]}")

        # Display the location, qualification, and experience requirements
        # st.markdown("### Requirements:")
        # for category, reqs in requirements.items():
        #     st.markdown(f"**{category}:**")
        #     for req in reqs:
        #         st.markdown(f"- {req}")

        if meta_data["query_type"] == "retrieve_applicant_jd":
            st.markdown(f"Using {meta_data['rag_mode']} to retrieve...")
            st.markdown(f"Returning top 5 most similar resumes...")

            button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
            for index, document in enumerate(document_list[:5]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            # st.markdown(f"**Extracted query**:\n`{meta_data['extracted_input']}`\n")
            # st.markdown(f"**Generated questions**:\n`{meta_data['subquestion_list']}`")
            # st.markdown(f"**Document re-ranking scores**:\n`{meta_data['retrieved_docs_with_scores']}`")

        elif meta_data["query_type"] == "retrieve_applicant_id":
            st.markdown(f"Using the ID to retrieve...")

            button_columns = st.columns([0.2, 0.2, 0.2, 0.2, 0.2], gap="small")
            for index, document in enumerate(document_list[:5]):
                with button_columns[index], st.popover(f"Resume {index + 1}"):
                    st.markdown(document)

            st.markdown(f"**Extracted query**:\n`{meta_data['extracted_input']}`\n")

if __name__ == "__main__":
    document_list = json.loads(sys.argv[1])
    meta_data = json.loads(sys.argv[2])
    time_elapsed = float(sys.argv[3])
    render(document_list, meta_data, time_elapsed)
