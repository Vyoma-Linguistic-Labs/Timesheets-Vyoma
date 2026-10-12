# Timesheet Generator (Streamlit App)

A Streamlit-based application that allows users to generate timesheets by selecting a date range and employee ID. The timesheet pulls data from ClickUp based on the employee's time entries and presents it in an Excel file for download. This app also allows users to open a Google sheet to check TS submission status.

## Features

- User-friendly interface for generating timesheets.
- Selectable start and end dates.
- Automatic conversion of time-tracked data from ClickUp into a neatly formatted Excel sheet.
- Displays total tracked hours per task and weekly totals.
- Includes a downloadable Excel report.
- Option to open a Google sheet to check TS submission status.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- Requests
- PIL (Pillow)
- ClickUp API
- Google Sheets (Optional)
- Urllib
- Webbrowser

![image](https://github.com/user-attachments/assets/10110a24-7f39-4407-9a32-620cfd3f6f61)

## Version
v4.0.0 (1st may 2025)

## Streamlit configuration

Run `streamlit run streamlit_TS.py`. For local use, create `.streamlit/secrets.toml` (excluded from Git):

```toml
clickup_api_key = "YOUR_CLICKUP_TOKEN"
team_id = "3314662"
```

For Streamlit Community Cloud, enter the same settings under Settings → Secrets. Use a Workspace Owner/Admin token with access to the required Spaces and tasks for employee-wide reporting. Tokens must not be committed to GitHub.
