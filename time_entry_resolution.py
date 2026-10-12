"""Recover task metadata omitted by ClickUp's bulk time-entry endpoint."""


class UnresolvedTaskError(ValueError):
    pass


def has_task(entry):
    task = entry.get("task")
    return isinstance(task, dict) and task.get("id") not in (None, "", "0", 0)


def resolve_time_entries(raw_entries, fetch_entry):
    """Keep original dates/durations and recover metadata by time-entry ID.

    Fail closed when a task cannot be recovered; never drop tracked hours.
    fetch_entry returns the JSON response from GET /time_entries/{entry_id}.
    """
    entries, unresolved, recovered = [], [], 0
    cache = {}
    for original in raw_entries:
        if int(original.get("duration") or 0) <= 0:
            continue
        entry = dict(original)
        if not has_task(entry):
            entry_id = entry.get("id")
            if entry_id:
                if entry_id not in cache:
                    response = fetch_entry(entry_id)
                    cache[entry_id] = response.get("data") if isinstance(response, dict) else None
                detail = cache[entry_id]
                if isinstance(detail, dict) and has_task(detail):
                    entry["task"] = detail["task"]
                    recovered += 1
            if not has_task(entry):
                unresolved.append(entry)
        entries.append(entry)
    if unresolved:
        hours = sum(int(e["duration"]) for e in unresolved) / 3600000
        raise UnresolvedTaskError(
            f"ClickUp task information could not be recovered for {len(unresolved)} "
            f"time entries ({hours:.2f} hours). No timesheet was generated. "
            "Check task access or retry; these entries have not been discarded."
        )
    return entries, recovered
