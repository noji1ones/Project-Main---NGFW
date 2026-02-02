import streamlit as st
import requests

st.title("AI NGFW Monitor")
st.header("Traffic Inspector")
#Input for dashboard to handle
input_data = st.text_input("Enter Packet Features (comma separated)", "80, 443, 6, 100")

if st.button("Analyze"):
    try:
        #Convert string "80, 443" to list [80.0, 443.0] 
        features_list = [float(x.strip()) for x in input_data.split(',') if x.strip()]

        #Working
        response = requests.post("http://api:8000/predict", json={"features": features_list})
        

        if response.status_code == 200:
            score = response.json()['malicious_score']
            if score > 0.5:
                st.error(f"THREAT DETECTED! Score: {score:.4f}")
            else:
                st.success(f"Traffic Good. Score: {score:.4f}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")
            
    except Exception as e:
        st.error(f"Connection Failed: {e}")