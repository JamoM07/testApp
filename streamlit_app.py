import streamlit as st

def get_user_input():
    st.header("Input Unit Numbers and Scenarios")

    # Input unit numbers
    unit_numbers = st.text_input("Enter unit numbers (separated by commas)", "")
    unit_numbers = [unit.strip() for unit in unit_numbers.split(",") if unit.strip()]

    # Input scenarios for each unit number
    st.subheader("Scenarios")

    num_strategies = st.selectbox("Number of Strategies", list(range(1, 11)), index=2)

    strategy_hours = {}
    for i in range(1, num_strategies + 1):
        strategy_name = f"Scenario {i}"
        strategy_hours[strategy_name] = st.number_input(f"{strategy_name} - Replacement Hours", min_value=0, value=20000)

    # File uploads
    st.subheader("File Uploads")
    ip19_data = st.file_uploader("Upload IP19 Data", type=["csv", "xlsx"])
    iw39_data = st.file_uploader("Upload IW39 Data", type=["csv", "xlsx"])
    ik17_component_data = st.file_uploader("Upload IK17 Component Data", type=["csv", "xlsx"])
    ik17_master_data = st.file_uploader("Upload IK17 Master Data", type=["csv", "xlsx"])

    unit_scenarios = {}
    for unit in unit_numbers:
        unit_scenarios[unit] = strategy_hours

    return unit_scenarios, ip19_data, iw39_data, ik17_component_data, ik17_master_data

def main():
    st.title("Your Web Application")

    # Get user input
    unit_scenarios, ip19_data, iw39_data, ik17_component_data, ik17_master_data = get_user_input()

    # Display the user input
    st.header("User Input")
    for unit, scenarios in unit_scenarios.items():
        st.write(f"Unit {unit}: {scenarios}")

    # File upload status
    if ip19_data:
        st.success("IP19 Data uploaded successfully!")
    if iw39_data:
        st.success("IW39 Data uploaded successfully!")
    if ik17_component_data:
        st.success("IK17 Component Data uploaded successfully!")
    if ik17_master_data:
        st.success("IK17 Master Data uploaded successfully!")

if __name__ == "__main__":
    main()
