# timesheet_generator_streamlit.py

# -*- coding: utf-8 -*-
"""
Created on Fri May 26 18:52:02 2023

Converted to Streamlit app on Fri, 4th Oct 2024.

@author: shank
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timezone, timedelta
import requests
import time
import numpy as np
import pytz
from PIL import Image
from io import BytesIO
import urllib.parse
import json
import calendar

api_key = st.secrets["clickup_api_key"]
team_id = st.secrets["team_id"]

__version__ = "v4.0.3"
__date__ = "2nd December 2025"
__auth__ = api_key

# Dictionary mapping month names to numbers
month_dict = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7,
    "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

# Define the MAIN AREA fields (Section 1 - At least one must be set)
main_area_fields = [
    "Linguistic Activity", "ELearning", "Technology", "Sales & Marketing", 
    "Customer Seva", "Finance", "Kids Persona", "Multi-Media", "HR & Admin",
    "Management Activities", "Outside Office Tasks", "Common Activities", "IKS"
]

# Define the OPTIONAL fields (Section 2)
optional_fields = [
    "Project ID", "Course", "Website", "Product", "Linguistic-Project"
]

# Combined list for backward compatibility
columns_to_check = main_area_fields + optional_fields

# Create a timezone object for IST
ist_timezone = pytz.timezone('Asia/Kolkata')

# Check if value is string 'nan' or np.nan
def is_nan(value):
    return value == 'nan' or (isinstance(value, float) and np.isnan(value))

def convert_milliseconds_to_hours_minutes(milliseconds):
    seconds = milliseconds / 1000
    minutes = seconds // 60
    hours = minutes // 60
    minutes = minutes % 60
    return (int(hours), int(minutes))

def clickup_get(url, params=None):
    """Call the ClickUp API and stop the app with a clear message if it fails."""
    headers = {"Content-Type": "application/json", "Authorization": __auth__}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
    except requests.RequestException as e:
        st.error(f"Could not reach ClickUp: {e}")
        st.stop()

    try:
        data = response.json()
    except ValueError:
        st.error(f"ClickUp returned an unexpected response (HTTP {response.status_code}).")
        st.stop()

    if response.status_code == 200:
        return data

    if response.status_code in (401, 403):
        if data.get('ECODE', '').startswith('OAUTH'):
            st.error(f"ClickUp rejected the API key ({data.get('err')}, {data.get('ECODE')}). "
                     "The token is invalid or revoked - generate a new one and update the app secrets.")
        else:
            st.error(f"ClickUp denied access ({data.get('err')}, {data.get('ECODE')}). "
                     "The token works but its owner lacks permission - use a token from a "
                     "Workspace Owner/Admin.")
    elif response.status_code == 429:
        st.error("ClickUp rate limit reached. Please wait a minute and try again.")
    else:
        st.error(f"ClickUp error (HTTP {response.status_code}): {data}")
    st.stop()

def memberInfo():
    data = clickup_get("https://api.clickup.com/api/v2/team")

    # Map last 4 characters of username (= Employee ID) -> ClickUp user id
    members_dict = {}
    for team in data.get('teams', []):
        if str(team.get('id')) != str(team_id):
            continue
        for member in team.get('members', []):
            user = member.get('user', {})
            username = user.get('username')
            if username:
                members_dict[username[-4:].upper()] = user['id']

    if not members_dict:
        st.error(f"The API token has no access to ClickUp team {team_id}, or the team has no members.")
        st.stop()

    return members_dict

def get_employee_name(employee_id):
    """Placeholder: returns a label built from the Employee ID."""
    return f"Employee_{employee_id}"

def dropdown_label(custom_field):
    """Return the option name for a dropdown custom field value (matched by orderindex)."""
    value = custom_field['value']
    options = custom_field.get('type_config', {}).get('options', [])
    for option in options:
        if option.get('orderindex') == value or option.get('id') == value:
            return option.get('name')
    if isinstance(value, int) and 0 <= value < len(options):
        return options[value].get('name')
    return None

def get_monthly_data(employee_key, month, year):
    """Get monthly timesheet data for an employee"""
    # Get first and last day of the month
    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1])
    
    # Convert to timestamps
    start_timestamp = int(datetime.combine(first_day, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())
    end_timestamp = int(datetime.combine(last_day, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())

    url = f"https://api.clickup.com/api/v2/team/{team_id}/time_entries"
    query = {
        "start_date": str((start_timestamp - 19800) * 1000),
        "end_date": str((end_timestamp + 86399) * 1000 - 19800000),
        "assignee": employee_key,
    }

    data = clickup_get(url, params=query)

    if 'data' not in data or not data['data']:
        return 0

    # Calculate total hours for the month (skip running timers, which have negative duration)
    total_milliseconds = sum(int(entry['duration']) for entry in data['data'] if int(entry['duration']) > 0)
    total_hours = total_milliseconds / 3600000  # Convert to hours
    
    return total_hours

# Optimized `get_selected_dates` function
def get_selected_dates(start_date, end_date, key, open_google_sheet, to_email, cc_email, ts_tracker_link, show_monthly_total=False, selected_month=None, selected_year=None):
    if not key:
        st.error("Please enter your Employee ID.")
        return

    key = key.upper()
        
    # Format dates
    start_date_str = start_date.strftime("%b %d")
    end_date_str = end_date.strftime("%b %d")
    year_str = str(start_date.year)
    
    st.session_state['start_date_str'] = start_date_str
    st.session_state['end_date_str'] = end_date_str
    st.session_state['year_str'] = year_str
    st.session_state['ts_tracker_link'] = ts_tracker_link
    
    # Generate filename
    filename = f"{key}_{start_date_str}_to_{end_date_str}_{year_str}.xlsx"

    # Retrieve information from ClickUp
    start_time_process = time.time()

    members_dict = memberInfo()
    if key not in members_dict:
        st.error("Invalid Employee ID. Please check and try again.")
        return

    employee_key = members_dict[key]

    # Convert start_date and end_date to timestamps
    start_timestamp = int(datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())
    end_timestamp = int(datetime.combine(end_date, datetime.min.time()).replace(tzinfo=timezone.utc).timestamp())

    url = f"https://api.clickup.com/api/v2/team/{team_id}/time_entries"
    query = {
        "start_date": str((start_timestamp - 19800) * 1000),  # Convert to milliseconds
        "end_date": str((end_timestamp + 86399) * 1000 - 19800000),
        "assignee": employee_key,
    }

    data = clickup_get(url, params=query)

    entries, skipped = [], []
    for e in data.get('data', []):
        duration = int(e.get('duration') or 0)
        if duration <= 0:          # running timer
            continue
        task = e.get('task')
        if isinstance(task, dict) and task.get('id'):
            entries.append(e)
        else:
            skipped.append(e)

    if skipped:
        hrs = sum(int(e['duration']) for e in skipped) / 3600000
        sample = skipped[0].get('task')
        st.warning(f"{len(skipped)} time entries ({hrs:.2f} h) are not linked to any task "
                   f"and were skipped. (Example task value: {type(sample).__name__} "
                   f"{str(sample)[:80]!r})")

    if not entries:
        st.error("No entries found in this date range. Please update entries in ClickUp.")
        return

    task_data = [
        {
            "Task Name": entry['task'].get('name', '0'),
            "Task ID": entry['task']['id'],
            "Task Status": (entry['task'].get('status') or {}).get('status', '0'),
            "Duration": int(entry['duration']),
            "Date": pd.Timestamp(int(entry['start']) // 1000, unit='s').date(),
            "Day": datetime.fromtimestamp(int(entry['start']) // 1000, tz=timezone.utc)
            .astimezone(ist_timezone)
            .strftime('%A'),
        }
        for entry in entries
    ]

    df = pd.DataFrame(task_data)

    # Optimize DataFrame operations
    days_of_week = ['Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df_pivot = df.pivot_table(index='Task ID', columns='Day', values='Duration', aggfunc='sum', fill_value=0)
    df_pivot = df_pivot.reindex(columns=days_of_week, fill_value=0)
    df_pivot = df_pivot.div(3600000).round(2)  # Convert milliseconds to hours

    df = df.drop(['Duration', 'Date', 'Day'], axis=1).drop_duplicates(subset='Task ID')
    df = df.merge(df_pivot, on='Task ID', how='left')
    
    ### OTHER Checks ###
    # iterate over the unique task IDs in the dataframe
    for task_id in df['Task ID'].unique():
        # fetch task details (stops with a clear message if ClickUp refuses)
        tasks = clickup_get(f"https://api.clickup.com/api/v2/task/{task_id}")

        hrs_mins = convert_milliseconds_to_hours_minutes(int(tasks.get('time_spent') or 0))
        df.loc[df['Task ID'] == task_id,
                 'Total (till date)'] = f"{hrs_mins[0]}h {hrs_mins[1]}m"
        # If there is no Custom field just continue
        try:
            for custom_field in tasks.get("custom_fields", []):
                # Process custom field logic
                if custom_field.get('value') is not None and custom_field.get('type') == 'drop_down':
                    label = dropdown_label(custom_field)
                    if label is not None:
                        df.loc[df['Task ID'] == task_id, custom_field['name']] = label
        except Exception as e:
            error_message = f"Error processing custom fields for task {task_id}: {e}"
            st.error(error_message)
            # Dump the response JSON nicely for debugging
            st.write("Task response:", json.dumps(tasks, indent=2))

    # Filter out holiday / leave rows (column name must match the one being filtered)
    if 'Common Activities' in df.columns:
        df = df[~df['Common Activities'].isin(['Vyoma Holiday', 'Personal Leave'])]
        if df.empty:
            st.error("Only holiday / leave entries were found in this date range.")
            return

    # Check if 'Goal Type' column exists
    if 'Goal Type' not in df.columns:
        # Add a new column with 'nan' values
        df['Goal Type'] = np.nan

    # Initialize lists to collect validation errors
    rows_missing_main_area = []
    row_id_missing_main_area = []
    rows_missing_goal_type = []
    row_id_missing_goal_type = []
    tasks_with_optional_but_no_main = []
    task_ids_with_optional_but_no_main = []
    
    # Get available columns in the dataframe
    available_main_area_fields = [field for field in main_area_fields if field in df.columns]
    available_optional_fields = [field for field in optional_fields if field in df.columns]
    
    # Iterate through rows in the DataFrame
    for index, row in df.iterrows():
        # Extract the 'Task Name' column value for the current row
        task_name = row['Task Name']
        task_id = row['Task ID']
        
        # Check if at least one Main Area field (Section 1) is set
        has_main_area = any(not is_nan(row[col]) for col in available_main_area_fields)
        
        # Check if any Optional field (Section 2) is set
        has_optional = any(not is_nan(row[col]) for col in available_optional_fields)
        
        # Validation 1: If no Main Area field is set at all
        if not has_main_area:
            rows_missing_main_area.append(task_name)
            row_id_missing_main_area.append(task_id)
        
        # Validation 2: If Optional field is set but no Main Area field is set
        if has_optional and not has_main_area:
            tasks_with_optional_but_no_main.append(task_name)
            task_ids_with_optional_but_no_main.append(task_id)
        
        # Validation 3: Goal Type is mandatory
        if is_nan(row['Goal Type']):
            rows_missing_goal_type.append(task_name)
            row_id_missing_goal_type.append(task_id)

    # Output validation errors
    if rows_missing_main_area or rows_missing_goal_type or tasks_with_optional_but_no_main:
        st.error("Some tasks are missing required information.")
        
        if rows_missing_main_area:
            st.write("**Main Area of Work (Section 1) is not set for the below task(s):**")
            st.write("At least one of these must be set: " + ", ".join(main_area_fields))
            for link_text, link_url in zip(rows_missing_main_area, row_id_missing_main_area):
                st.write(f"[{link_text}](https://app.clickup.com/t/{link_url})")
        
        if tasks_with_optional_but_no_main:
            st.error("**Some tasks have Optional fields (Section 2) set but no Main Area field (Section 1) set:**")
            st.write("When you set any Optional field (Project ID, Course, Website, Product, Linguistic-Project), you must also set at least one Main Area field.")
            for task_name, task_id in zip(tasks_with_optional_but_no_main, task_ids_with_optional_but_no_main):
                st.write(f"[{task_name}](https://app.clickup.com/t/{task_id})")
        
        if rows_missing_goal_type:
            st.write("**Goal Type (Mandatory Field) not set for:**")
            for link_text, link_url in zip(rows_missing_goal_type, row_id_missing_goal_type):
                st.write(f"[{link_text}](https://app.clickup.com/t/{link_url})")
        
        st.info("You can try generating your timesheet again once you set the above information in these tasks.")
        return
    
    # Add total tracked time for the week
    df['Total (this week)'] = df[days_of_week].sum(axis=1)

    # Add totals row
    totals = df[days_of_week].sum(axis=0)
    df = pd.concat([df, totals.to_frame().T], ignore_index=True)
    weekly_total = df.iloc[-1].sum()
    st.session_state['weekly_total'] = weekly_total
    df.at[df.index[-1], 'Task Status'] = 'Daily Totals ->'
    empty_row = pd.Series([np.nan] * len(df.columns), index=df.columns)
    df = pd.concat([df, empty_row.to_frame().T], ignore_index=True)
    df.at[df.index[-1], 'Total (this week)'] = weekly_total

    # Add monthly total if requested
    if show_monthly_total and selected_month and selected_year:
        monthly_total = get_monthly_data(employee_key, selected_month, selected_year)
        st.session_state['monthly_total'] = monthly_total
        empty_row = pd.Series([np.nan] * len(df.columns), index=df.columns)
        df = pd.concat([df, empty_row.to_frame().T], ignore_index=True)
        df.at[df.index[-1], 'Task Status'] = f'Total Hours for {calendar.month_name[selected_month]} {selected_year}'
        df.at[df.index[-1], 'Total (this week)'] = monthly_total

    days_diff = (end_date - start_date).days + 1
    summary_row_index = df.index[-2 if show_monthly_total else -1]
    if days_diff <= 7:
        week_number = end_date.isocalendar()[1]
        st.session_state['week_number'] = week_number
        df.at[summary_row_index, 'Task Status'] = "Week's total ="
        df.at[summary_row_index, 'Task Name'] = f'Week #{week_number} - {start_date_str}, {year_str} - {end_date_str}, {year_str}'
    else:
        df.at[summary_row_index, 'Task Status'] = 'Total Hours'
        df.at[summary_row_index, 'Task Name'] = f'{start_date_str}, {year_str} - {end_date_str}, {year_str}'

    # Reorder column: move 'Total (this week)' to the 11th position.
    df.insert(10, 'Total (this week)', df.pop('Total (this week)'))

    # Write to Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        worksheet = writer.sheets['Sheet1']
        for row_num, value in enumerate(df['Task ID'], start=1):
            if pd.isna(value):
                break
            worksheet.write_url(row_num, df.columns.get_loc('Task ID'), f'https://app.clickup.com/t/{value}', string=value)

    processed_data = output.getvalue()
    st.success(f"Successfully generated the timesheet: {filename}")
    st.download_button(
        label="Download Excel File",
        data=processed_data,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.write(f"Total Hours for this time frame: {weekly_total:.2f}")
    if show_monthly_total and selected_month and selected_year:
        st.write(f"Total Hours for {calendar.month_name[selected_month]} {selected_year}: {monthly_total:.2f}")
    st.write(f"Processing Time: {time.time() - start_time_process:.2f} seconds")    
    
    if open_google_sheet:
        st.markdown("""
        <script>
        window.open('https://docs.google.com/spreadsheets/d/1XLDSTT5m952eiOXhiUtxldIIoEAfQgiVKv5XY2HFOBg/edit?usp=sharing', '_blank');
        </script>
        """, unsafe_allow_html=True)
        st.success("Opening Google Sheet for TS Submission Status...")
        st.write("If the sheet didn't open automatically, [click here to open it manually](https://docs.google.com/spreadsheets/d/1XLDSTT5m952eiOXhiUtxldIIoEAfQgiVKv5XY2HFOBg/edit?usp=sharing)")
    
    # 1) replace <NA> with blanks
    df = df.fillna('')    
    df = df.replace(['nan', 'na'], '', regex=True)
    
    # Reset index to start from 1 instead of 0, but only for actual task rows
    df.reset_index(drop=True, inplace=True)
    
    # Find the index where totals start (where 'Task Status' is 'Daily Totals ->')
    totals_start_index = df[df['Task Status'] == 'Daily Totals ->'].index
    
    if len(totals_start_index) > 0:
        # Only reset index for rows before the totals
        actual_task_rows = totals_start_index[0]
        
        # Create a new index where task rows start from 1 and totals rows are empty
        new_index = []
        for i in range(len(df)):
            if i < actual_task_rows:
                new_index.append(str(i + 1))  # Convert to string for consistency
            else:
                new_index.append("")  # Empty string for totals rows
        
        df.index = new_index
    else:
        # Fallback: set index to start from 1 for all rows
        df.index = [str(i + 1) for i in range(len(df))]
    
    return df

# Function to calculate the default start and end dates based on today's date
def calculate_default_dates():
    today = datetime.today()

    # Calculate the previous Friday
    days_to_friday = (today.weekday() - 4) % 7  # Friday is the 4th day in Python's weekday system
    end_date = today - timedelta(days=days_to_friday)
    
    # If today is Friday, set end date to today
    if today.weekday() == 4:
        end_date = today

    # Calculate the Saturday before the selected Friday
    start_date = end_date - timedelta(days=6)  # Saturday is one day before Friday

    return start_date, end_date
    
# Streamlit UI
def main():
    st.set_page_config(page_title="Timesheet Generator", page_icon=":calendar:", layout="wide")
    
    # Create two columns: one for the logo and one for the title
    col1, col2 = st.columns([1, 2])  # Adjust the column ratios as needed

    # Display Image in the first column
    with col1:
        image_url = "https://digitalsanskritguru.com/wp-content/uploads/2020/05/Vyoma_Logo_Blue_500x243.png"
        response = requests.get(image_url)
        image_data = response.content
        image = Image.open(BytesIO(image_data))
        image = image.resize((167, 81))
        st.image(image, width="content")

    # Display Title in the second column (centered)
    with col2:
        st.markdown("<h1 style='text-align: left;'>Timesheet Generator</h1>", unsafe_allow_html=True)

    # Add instructions at the top
    st.markdown("---")
    st.markdown("### 📋 Instructions for Setting Task Fields in ClickUp")
    st.info("""
    **While setting various fields in your tasks, please ensure the following:**
    
    **1. Main Area of Work (Section 1) - At least ONE must be set for every task:**
    - Linguistic Activity, ELearning, Technology, Sales & Marketing, Customer Seva, Finance, 
    Kids Persona, Multi-Media, Management Activities, Outside Office Tasks, Common Activities, IKS
    
    **2. Optional Fields (Section 2) - Can be set if the task is related to any of these:**
    - Project ID, Course, Website, Product, Linguistic-Project
    - **Note:** If you set any field in Section 2, at least one field in Section 1 must also be set
    
    **3. Mandatory Field (Section 3) - Must be set for ALL tasks:**
    - Goal Type
    """)
    st.markdown("---")

    # Initialize session state variables
    if "timesheet" not in st.session_state:
        st.session_state["timesheet"] = None
        st.session_state['start_date_str'] = None
        st.session_state['end_date_str'] = None
        st.session_state['year_str'] = None
        st.session_state['week_number'] = None
        st.session_state['ts_tracker_link'] = None
        st.session_state['monthly_total'] = None
    if "submit_clicked" not in st.session_state:
        st.session_state["submit_clicked"] = False
    if "download_clicked" not in st.session_state:
        st.session_state["download_clicked"] = False

    # Employee Key Entry
    key = st.text_input("Employee ID: (e.g., C047)")
    
    # TS Tracker Link - Mandatory field
    ts_tracker_link = st.text_input("TS Tracker Link: (Mandatory)", placeholder="Enter your timesheet tracker link")
    
    # Get default start and end dates based on today's date
    default_start_date, default_end_date = calculate_default_dates()

    # Date Selection with Display in 'Day, dd Mon yyyy' Format after Selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", default_start_date)
        # Format the selected start date
        formatted_start_date = start_date.strftime('%a, %d %b %Y')
        # Display the formatted start date just below the selection
        st.write(f"Selected Start Date: {formatted_start_date}")
        to_email = st.text_input("To (Recipient Email Address)")

    with col2:
        end_date = st.date_input("End Date", default_end_date)
        # Format the selected end date
        formatted_end_date = end_date.strftime('%a, %d %b %Y')
        # Display the formatted end date just below the selection
        st.write(f"Selected End Date: {formatted_end_date}")
        cc_email = st.text_input("CC (CC Email Address)", 
                                 "srilatha.vyoma@gmail.com, hr@vyomalabs.in")  # Prefill CC field    

    # Monthly total options
    show_monthly_total = st.checkbox("Show Monthly Total")
    
    selected_month = None
    selected_year = None
    
    if show_monthly_total:
        col1, col2 = st.columns(2)
        with col1:
            selected_month = st.selectbox("Select Month", 
                                        options=list(range(1, 13)),
                                        format_func=lambda x: calendar.month_name[x],
                                        index=datetime.now().month - 1)
        with col2:
            selected_year = st.selectbox("Select Year", 
                                       options=list(range(2020, 2030)),
                                       index=datetime.now().year - 2020)

    # Create checkboxes
    open_google_sheet = st.checkbox("Open the Google sheet for TS Submission Status")    

    # Button to generate the timesheet
    if st.button("Generate Timesheet"):
        if not ts_tracker_link.strip():
            st.error("TS Tracker Link is mandatory. Please enter your timesheet tracker link.")
        else:
            # Generate the DataFrame and store it in session state
            st.session_state["timesheet"] = get_selected_dates(start_date, end_date, key, open_google_sheet, 
                                                             to_email, cc_email, ts_tracker_link, 
                                                             show_monthly_total, selected_month, selected_year)        
        
    # Check if the timesheet exists in session state
    if st.session_state["timesheet"] is not None:
        # Add Vyoma logo above the table
        st.markdown("---")
        
        # Display logo above the table
        col_logo, col_spacer = st.columns([1, 4])
        with col_logo:
            image_url = "https://digitalsanskritguru.com/wp-content/uploads/2020/05/Vyoma_Logo_Blue_500x243.png"
            response = requests.get(image_url)
            image_data = response.content
            image = Image.open(BytesIO(image_data))
            image = image.resize((100, 49))  # Smaller size for table header
            st.image(image, width="content")
        
        # Display the timesheet as a table   
        # cell‐wise formatter
        def fmt_cell(x):
            if isinstance(x, (int, float, np.floating, np.integer)):
                return f"{x:.2f}"
            return x
        # 3) build the Styler
        styled = (
            st.session_state["timesheet"].fillna('')  # blank out the <NA>s
              .style
              .format(fmt_cell)                # apply fmt_cell to each cell
               .set_table_attributes('style="width:100%; table-layout: auto; font-size: .75vw"')
               .to_html()
        )
        st.markdown(styled, unsafe_allow_html=True)        
        
        st.session_state["submit_clicked"] = False
        st.session_state["download_clicked"] = False
        st.success("Timesheet generated successfully!")
        
        # Checkbox for confirmation that user has pasted data in tracker
        ts_pasted_confirmation = st.checkbox("Have you pasted this week's timesheet data in your online tracker?", 
                                           key="ts_confirmation")
        
        if ts_pasted_confirmation:
            employee_name = get_employee_name(key)
            subject = f"Timesheet for Week #{st.session_state['week_number']} - {st.session_state['start_date_str']}, {st.session_state['year_str']} - {st.session_state['end_date_str']}, {st.session_state['year_str']}"
            
            # Prepare the subject and body with hyperlinked tracker    
            subject_encoded = urllib.parse.quote(subject.encode('utf-8'))
            body = f"Ram ram ram,\nPlease find my weekly timesheet in this Google tracker -\n\n{employee_name} TS Tracker: {st.session_state['ts_tracker_link']}\n\n<Paste the timesheet screenshot here>\n"
            body_encoded = urllib.parse.quote(body.encode('utf-8'))

            # Construct the Gmail URL with the recipient, subject, body, and CC fields pre-filled
            gmail_url = f"https://mail.google.com/mail/?view=cm&to={to_email}&cc={cc_email}&su={subject_encoded}&body={body_encoded}"
            
            st.markdown(f'''
                <a href="{gmail_url}" target="_blank">
                    <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;">
                        Click Here to Send Email
                    </button>
                </a>
            ''', unsafe_allow_html=True)
        else:
            if st.session_state["timesheet"] is not None:
                st.warning("Please confirm that you have pasted the timesheet data in your tracker before sending the email.")
                                                    
    else:
        st.info("Click 'Generate Timesheet' to create a timesheet.")

    # Footer
    st.write(f"Version {__version__} {__date__}")

if __name__ == "__main__":
    main()