"""Polling page: the poll form on the left, its completion analysis on the right."""

import streamlit as st

from pulse_ui.components import tables, ranking, poll_form
from pulse_ui.components.shortcuts import activate_shortcuts

HEIGHT = 800

if st.query_params.get("debug") == "True":
    st.sidebar.write(poll_form.draft_poll())

task_col, comp_col = st.columns(2)
task_col.subheader("Create a poll")
comp_col.subheader("Completion analysis")
task_cont = task_col.container(border=True, height=HEIGHT)
comp_cont = comp_col.container(border=True, height=HEIGHT)

with task_cont:
    poll_form.selector()
    poll_form.prompts()  # fragment
with task_cont.container(border=True):
    tables.completions()  # fragment

ranking.show(comp_cont)

activate_shortcuts(fn_map={"save": poll_form.save, "rank": ranking.request, "run": poll_form.run})
_, cap = comp_col.columns((2.75, 1))
cap.caption("Press Ctrl+K to open the legend")
