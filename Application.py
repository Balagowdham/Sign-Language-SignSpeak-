# SignSpeak - Real-Time ASL Fingerspelling Recognizer
# GUI application: captures webcam frames, extracts a hand region, runs it
# through the trained CNN models, and builds letters -> words -> sentences.
#
# NOTE: Model files, dataset folders, and the underlying prediction approach
# are unchanged from the original project this is adapted from. Only the
# interface (layout, styling, structure) has been redesigned.

import os
import operator
from string import ascii_uppercase

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from spellchecker import SpellChecker

# NOTE: model_from_json + a plain-named .h5 weights file (the format this
# project's trained models were saved in) is Keras 2's "legacy" save format.
# Modern `pip install keras` installs Keras 3 by default, whose top-level
# `keras.models.model_from_json` cannot deserialize configs written by
# Keras 2 and will raise `TypeError: Could not locate class 'Sequential'`.
# `tf_keras` is Google's maintained Keras-2-compatible package and loads
# these files correctly - install it alongside tensorflow (see
# requirements.txt) rather than switching this import back to `keras`.
from tf_keras.models import model_from_json

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------
BG_DARK = "#141821"
BG_PANEL = "#1d2330"
ACCENT = "#3ddc97"
ACCENT_DIM = "#2a9d76"
TEXT_MAIN = "#eef1f6"
TEXT_MUTED = "#8b93a7"
FONT_HEAD = ("Segoe UI", 20, "bold")
FONT_LABEL = ("Segoe UI", 13, "bold")
FONT_VALUE = ("Segoe UI", 16)
FONT_SUGGEST = ("Segoe UI", 12)

MODEL_DIR = "Models"


class SignSpeakApp:
    """Main application window."""

    def __init__(self):
        self._load_models()
        self._init_state()
        self._build_window()
        self._build_layout()
        self.video_loop()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _load_models(self):
        self.spell = SpellChecker()
        self.capture = cv2.VideoCapture(0)

        self.model_main = self._load_one(
            os.path.join(MODEL_DIR, "model_new.json"),
            os.path.join(MODEL_DIR, "model_new.h5"),
        )
        self.model_dru = self._load_one(
            os.path.join(MODEL_DIR, "model-bw_dru.json"),
            os.path.join(MODEL_DIR, "model-bw_dru.h5"),
        )
        self.model_tkdi = self._load_one(
            os.path.join(MODEL_DIR, "model-bw_tkdi.json"),
            os.path.join(MODEL_DIR, "model-bw_tkdi.h5"),
        )
        self.model_smn = self._load_one(
            os.path.join(MODEL_DIR, "model-bw_smn.json"),
            os.path.join(MODEL_DIR, "model-bw_smn.h5"),
        )
        print("Models loaded.")

    @staticmethod
    def _load_one(json_path, weights_path):
        with open(json_path, "r") as f:
            model = model_from_json(f.read())
        model.load_weights(weights_path)
        return model

    def _init_state(self):
        self.letter_counts = {"blank": 0}
        for letter in ascii_uppercase:
            self.letter_counts[letter] = 0

        self.blank_seen = False
        self.current_symbol = "-"
        self.current_word = ""
        self.current_sentence = ""

    # ------------------------------------------------------------------
    # Window / layout
    # ------------------------------------------------------------------
    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("SignSpeak — Real-Time ASL Fingerspelling")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1040x680")
        self.root.minsize(940, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Suggest.TButton",
            font=FONT_SUGGEST,
            background=BG_PANEL,
            foreground=TEXT_MAIN,
            borderwidth=0,
            padding=8,
        )
        style.map("Suggest.TButton", background=[("active", ACCENT_DIM)])

    def _build_layout(self):
        # Header bar
        header = tk.Frame(self.root, bg=BG_DARK)
        header.pack(fill="x", padx=20, pady=(16, 8))

        tk.Label(
            header, text="SignSpeak", font=FONT_HEAD, bg=BG_DARK, fg=ACCENT
        ).pack(side="left")
        tk.Label(
            header,
            text="  live ASL fingerspelling → text",
            font=("Segoe UI", 12),
            bg=BG_DARK,
            fg=TEXT_MUTED,
        ).pack(side="left", padx=(4, 0))

        # Body: left = video feeds, right = output panel
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_video_column(body)
        self._build_output_column(body)

    def _build_video_column(self, parent):
        col = tk.Frame(parent, bg=BG_DARK)
        col.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        cam_card = self._card(col, "Camera")
        cam_card.pack(fill="both", expand=True, pady=(0, 12))
        self.camera_panel = tk.Label(cam_card, bg="black")
        self.camera_panel.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        mask_card = self._card(col, "Processed hand region")
        mask_card.pack(fill="x")
        self.mask_panel = tk.Label(mask_card, bg="black")
        self.mask_panel.pack(padx=10, pady=(0, 10))

    def _build_output_column(self, parent):
        col = tk.Frame(parent, bg=BG_DARK)
        col.grid(row=0, column=1, sticky="nsew")

        # Current letter
        letter_card = self._card(col, "Detected letter")
        letter_card.pack(fill="x", pady=(0, 12))
        self.letter_value = tk.Label(
            letter_card, text="-", font=("Segoe UI", 42, "bold"),
            bg=BG_PANEL, fg=ACCENT,
        )
        self.letter_value.pack(pady=(0, 8))

        # Word
        word_card = self._card(col, "Current word")
        word_card.pack(fill="x", pady=(0, 12))
        self.word_value = tk.Label(
            word_card, text="", font=FONT_VALUE, bg=BG_PANEL, fg=TEXT_MAIN,
            anchor="w", justify="left", wraplength=320,
        )
        self.word_value.pack(fill="x")

        # Sentence
        sentence_card = self._card(col, "Sentence")
        sentence_card.pack(fill="both", expand=True, pady=(0, 12))
        self.sentence_value = tk.Label(
            sentence_card, text="", font=FONT_VALUE, bg=BG_PANEL, fg=TEXT_MAIN,
            anchor="nw", justify="left", wraplength=320,
        )
        self.sentence_value.pack(fill="both", expand=True)

        # Suggestions
        suggest_card = self._card(col, "Word suggestions")
        suggest_card.pack(fill="x")
        row = tk.Frame(suggest_card, bg=BG_PANEL)
        row.pack(fill="x")
        self.suggest_buttons = [
            ttk.Button(row, text="", style="Suggest.TButton",
                       command=lambda i=i: self.accept_suggestion(i))
            for i in range(3)
        ]
        for b in self.suggest_buttons:
            b.pack(side="left", expand=True, fill="x", padx=4, pady=(0, 8))

    def _card(self, parent, title):
        card = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=10)
        tk.Label(
            card, text=title.upper(), font=FONT_LABEL, bg=BG_PANEL, fg=TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 6))
        return card

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def video_loop(self):
        ok, frame = self.capture.read()

        if ok:
            frame = cv2.flip(frame, 1)

            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (61, 220, 151), 2)

            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self._show_on_panel(self.camera_panel, rgba)

            roi = rgba[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            adaptive = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            _, mask = cv2.threshold(
                adaptive, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            self.predict(mask)
            self._show_on_panel(self.mask_panel, mask)

            self.letter_value.config(text=self.current_symbol)
            self.word_value.config(text=self.current_word or " ")
            self.sentence_value.config(text=self.current_sentence or " ")

            suggestions = self.get_suggestions(self.current_word)
            for i, btn in enumerate(self.suggest_buttons):
                btn.config(text=suggestions[i] if i < len(suggestions) else "")

        self.root.after(10, self.video_loop)

    @staticmethod
    def _show_on_panel(panel, image_array):
        img = Image.fromarray(image_array)
        photo = ImageTk.PhotoImage(image=img)
        panel.imgtk = photo
        panel.config(image=photo)

    # ------------------------------------------------------------------
    # Prediction (same two-stage approach as the original model)
    # ------------------------------------------------------------------
    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        inp = test_image.reshape(1, 128, 128, 1)

        result_main = self.model_main.predict(inp, verbose=0)
        result_dru = self.model_dru.predict(inp, verbose=0)
        result_tkdi = self.model_tkdi.predict(inp, verbose=0)
        result_smn = self.model_smn.predict(inp, verbose=0)

        # Stage 1: full alphabet + blank
        scores = {"blank": result_main[0][0]}
        for idx, letter in enumerate(ascii_uppercase, start=1):
            scores[letter] = result_main[0][idx]
        ranked = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        symbol = ranked[0][0]

        # Stage 2: disambiguate visually similar letter groups
        if symbol in ("D", "R", "U"):
            group = {"D": result_dru[0][0], "R": result_dru[0][1], "U": result_dru[0][2]}
            symbol = max(group.items(), key=operator.itemgetter(1))[0]

        if symbol in ("D", "I", "K", "T"):
            group = {
                "D": result_tkdi[0][0], "I": result_tkdi[0][1],
                "K": result_tkdi[0][2], "T": result_tkdi[0][3],
            }
            symbol = max(group.items(), key=operator.itemgetter(1))[0]

        if symbol in ("M", "N", "S"):
            group = {"M": result_smn[0][0], "N": result_smn[0][1], "S": result_smn[0][2]}
            best = max(group.items(), key=operator.itemgetter(1))[0]
            if best == "S":
                symbol = best
            # otherwise keep the stage-1 symbol

        self.current_symbol = symbol
        self._accumulate(symbol)

    def _accumulate(self, symbol):
        if symbol == "blank":
            for letter in ascii_uppercase:
                self.letter_counts[letter] = 0

        self.letter_counts[symbol] += 1

        if self.letter_counts[symbol] <= 60:
            return

        # Require the winning symbol to be clearly dominant before committing it
        for letter in ascii_uppercase:
            if letter == symbol:
                continue
            if abs(self.letter_counts[symbol] - self.letter_counts[letter]) <= 20:
                self._reset_counts()
                return

        self._reset_counts()

        if symbol == "blank":
            if not self.blank_seen:
                self.blank_seen = True
                if self.current_sentence:
                    self.current_sentence += " "
                self.current_sentence += self.current_word
                self.current_word = ""
        else:
            if len(self.current_sentence) > 16:
                self.current_sentence = ""
            self.blank_seen = False
            self.current_word += symbol

    def _reset_counts(self):
        self.letter_counts["blank"] = 0
        for letter in ascii_uppercase:
            self.letter_counts[letter] = 0

    # ------------------------------------------------------------------
    # UI actions
    # ------------------------------------------------------------------
    def get_suggestions(self, word, limit=3):
        """Return up to `limit` likely word completions, most common first."""
        if not word:
            return []
        candidates = self.spell.candidates(word) or set()
        ranked = sorted(candidates, key=self.spell.word_usage_frequency, reverse=True)
        return ranked[:limit]

    def accept_suggestion(self, index):
        suggestions = self.get_suggestions(self.current_word)
        if index < len(suggestions):
            self.current_word = ""
            if self.current_sentence:
                self.current_sentence += " "
            self.current_sentence += suggestions[index]

    def destructor(self):
        print("Closing SignSpeak...")
        self.root.destroy()
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting SignSpeak...")
    app = SignSpeakApp()
    app.root.mainloop()
