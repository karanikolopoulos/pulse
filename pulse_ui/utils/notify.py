"""Toasts that survive `st.rerun()`: queue them now, show them at the start of the next run."""

import streamlit as st

from pulse_ui.session import notices


def notify(message: str) -> None:
    notices.pending = [*(notices.pending or []), message]


def show_pending() -> None:
    for message in notices.pending or []:
        st.toast(message)
    del notices.pending
