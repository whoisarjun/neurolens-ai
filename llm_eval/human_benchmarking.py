import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
import json
import cache

CACHE_DIR = Path('cache/transcripts/')
EVAL_DIR = Path('EVAL_DATA')

# Dark mode color scheme

COLORS = {
    'bg': '#1e1e1e',
    'card': '#2d2d2d',
    'text': '#e0e0e0',
    'text_dim': '#a0a0a0',
    'accent': '#0078d4',
    'accent_hover': '#1e88e5',
    'success': '#28a745',
    'warning': '#ffc107',
    'border': '#404040',
    'input_bg': '#383838',
    'slider_track': '#505050',
    'slider_active': '#0078d4'
}

# Font sizes (bigger / more readable)
FONTS = {
    'xs': 16,
    'sm': 18,
    'md': 20,
    'lg': 24,
    'xl': 28,
    'xxl': 36
}


class BenchmarkUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Transcript Benchmarking Tool")

        # Get screen size and set window to fit laptop screen
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Use 95% of screen to fit comfortably
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.92)

        # Center window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.configure(bg=COLORS['bg'])

        # Load features
        try:
            with open('features.json', 'r') as f:
                features_data = json.load(f)
                self.features = features_data['features']
        except Exception as e:
            messagebox.showerror("Error", f"Could not load features.json: {e}")
            self.root.destroy()
            return

        # Get user name with dark dialog
        self.username = self.get_username()
        if not self.username:
            self.root.destroy()
            return

        self.save_file = Path(f"{self.username.lower()}.json")

        # Load audio files
        self.audio_files = sorted(list(EVAL_DIR.glob("*.wav")))
        if not self.audio_files:
            messagebox.showerror("Error", "No audio files found in EVAL_DATA folder")
            self.root.destroy()
            return

        # Load existing progress
        self.data = self.load_progress()

        # State
        self.current_file_idx = self.find_next_incomplete()
        self.current_scores = [2] * len(self.features)
        self.current_feature_idx = 0

        # Setup UI
        self.setup_ui()
        self.load_current_file()

        # Keybindings
        self.root.bind('z', lambda e: self.prev_feature())
        self.root.bind('x', lambda e: self.next_feature())
        self.root.bind('<Return>', lambda e: self.next_feature())

    def get_username(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Username")
        dialog.configure(bg=COLORS['bg'])
        dialog.geometry("400x150")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Enter your name:", font=('Segoe UI', FONTS['lg']),
                 bg=COLORS['bg'], fg=COLORS['text']).pack(pady=20)

        entry = tk.Entry(dialog, font=('Segoe UI', FONTS['md']), bg=COLORS['input_bg'],
                         fg=COLORS['text'], insertbackground=COLORS['text'], bd=0, relief=tk.FLAT)
        entry.pack(padx=40, pady=10, fill=tk.X, ipady=8)
        entry.focus()

        result = {'name': None}

        def on_submit():
            result['name'] = entry.get().strip()
            dialog.destroy()

        entry.bind('<Return>', lambda e: on_submit())

        btn = tk.Button(dialog, text="Start", command=on_submit, font=('Segoe UI', FONTS['md'], 'bold'),
                        bg=COLORS['accent'], fg='white', bd=0, relief=tk.FLAT,
                        padx=30, pady=10, cursor='hand2')
        btn.pack(pady=10)

        dialog.wait_window()
        return result['name']

    def load_progress(self):
        if self.save_file.exists():
            try:
                with open(self.save_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"results": []}

    def find_next_incomplete(self):
        completed_files = {r['file'] for r in self.data['results']}
        for idx, fp in enumerate(self.audio_files):
            if str(fp) not in completed_files:
                return idx
        return len(self.audio_files) - 1

    def setup_ui(self):
        # Main container - tighter spacing
        main_frame = tk.Frame(self.root, bg=COLORS['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Progress section - compact
        progress_frame = tk.Frame(main_frame, bg=COLORS['card'], highlightbackground=COLORS['border'],
                                  highlightthickness=1)
        progress_frame.pack(fill=tk.X, pady=(0, 6))

        progress_inner = tk.Frame(progress_frame, bg=COLORS['card'])
        progress_inner.pack(padx=10, pady=6, fill=tk.X)

        # Progress labels
        progress_top = tk.Frame(progress_inner, bg=COLORS['card'])
        progress_top.pack(fill=tk.X, pady=(0, 3))

        self.progress_pct = tk.Label(progress_top, text="0%", font=('Segoe UI', FONTS['md'], 'bold'),
                                     bg=COLORS['card'], fg=COLORS['text'])
        self.progress_pct.pack(side=tk.LEFT)

        self.progress_count = tk.Label(progress_top, text="0/150", font=('Segoe UI', FONTS['md'], 'bold'),
                                       bg=COLORS['card'], fg=COLORS['text'])
        self.progress_count.pack(side=tk.RIGHT)

        # Progress bar
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=COLORS['slider_track'],
                        background=COLORS['success'],
                        borderwidth=0,
                        thickness=18)

        # Button styles (ttk) so colors work reliably on macOS
        style.configure('Nav.TButton',
                        font=('Segoe UI', FONTS['md'], 'bold'),
                        padding=(18, 10),
                        background=COLORS['input_bg'],
                        foreground=COLORS['accent'])
        style.map('Nav.TButton',
                  background=[('active', COLORS['input_bg']), ('disabled', COLORS['input_bg'])],
                  foreground=[('disabled', COLORS['text_dim'])])

        style.configure('Complete.TButton',
                        font=('Segoe UI', FONTS['md'], 'bold'),
                        padding=(18, 10),
                        background=COLORS['success'],
                        foreground='white')
        style.map('Complete.TButton',
                  background=[('active', COLORS['success']), ('!disabled', COLORS['success'])],
                  foreground=[('active', 'white'), ('!disabled', 'white')])

        self.progress_bar = ttk.Progressbar(progress_inner, mode='determinate',
                                            style="Custom.Horizontal.TProgressbar")
        self.progress_bar.pack(fill=tk.X)

        # File info - compact
        info_frame = tk.Frame(main_frame, bg=COLORS['card'], highlightbackground=COLORS['border'],
                              highlightthickness=1)
        info_frame.pack(fill=tk.X, pady=(0, 6))

        info_inner = tk.Frame(info_frame, bg=COLORS['card'])
        info_inner.pack(padx=10, pady=6)

        tk.Label(info_inner, text="FILE:", font=('Segoe UI', FONTS['sm'], 'bold'),
                 bg=COLORS['card'], fg=COLORS['text_dim']).pack(side=tk.LEFT, padx=(0, 8))

        self.file_label = tk.Label(info_inner, text="", font=('Segoe UI', FONTS['lg'], 'bold'),
                                   bg=COLORS['card'], fg=COLORS['text'])
        self.file_label.pack(side=tk.LEFT)

        # Transcript - larger text for readability
        transcript_frame = tk.Frame(main_frame, bg=COLORS['card'], highlightbackground=COLORS['border'],
                                    highlightthickness=1)
        transcript_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        transcript_inner = tk.Frame(transcript_frame, bg=COLORS['card'])
        transcript_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        tk.Label(transcript_inner, text="TRANSCRIPT:", font=('Segoe UI', FONTS['sm'], 'bold'),
                 bg=COLORS['card'], fg=COLORS['text_dim']).pack(anchor=tk.W)

        transcript_scroll_frame = tk.Frame(transcript_inner, bg=COLORS['card'])
        transcript_scroll_frame.pack(fill=tk.BOTH, expand=True, pady=(3, 0))

        scrollbar = tk.Scrollbar(transcript_scroll_frame, bg=COLORS['card'], troughcolor=COLORS['input_bg'])
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # LARGER FONT for transcript readability
        self.transcript_text = tk.Text(transcript_scroll_frame, wrap=tk.WORD,
                                       font=('Segoe UI', FONTS['lg']), height=8,
                                       bg=COLORS['input_bg'], fg=COLORS['text'],
                                       relief=tk.FLAT, bd=0, padx=12, pady=10,
                                       yscrollcommand=scrollbar.set,
                                       insertbackground=COLORS['text'])
        self.transcript_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.transcript_text.yview)
        self.transcript_text.config(state=tk.DISABLED)

        # Feature scoring
        feature_frame = tk.Frame(main_frame, bg=COLORS['card'], highlightbackground=COLORS['border'],
                                 highlightthickness=1)
        feature_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        feature_inner = tk.Frame(feature_frame, bg=COLORS['card'])
        feature_inner.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        self.feature_name = tk.Label(feature_inner, text="", font=('Segoe UI', FONTS['xl'], 'bold'),
                                     bg=COLORS['card'], fg=COLORS['text'])
        self.feature_name.pack(anchor=tk.W)

        self.feature_desc = tk.Label(feature_inner, text="", font=('Segoe UI', FONTS['md']),
                                     bg=COLORS['card'], fg=COLORS['text_dim'],
                                     wraplength=1200, justify=tk.LEFT)
        self.feature_desc.pack(anchor=tk.W, pady=(2, 6))

        # Scale options in horizontal grid
        tk.Label(feature_inner, text="SCORING SCALE:", font=('Segoe UI', FONTS['md'], 'bold'),
                 bg=COLORS['card'], fg=COLORS['text_dim']).pack(anchor=tk.W, pady=(0, 3))

        # Horizontal grid for scale options
        self.scale_options_frame = tk.Frame(feature_inner, bg=COLORS['card'])
        self.scale_options_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        # Slider section
        tk.Label(feature_inner, text="YOUR SCORE:", font=('Segoe UI', FONTS['md'], 'bold'),
                 bg=COLORS['card'], fg=COLORS['text_dim']).pack(anchor=tk.W, pady=(3, 2))

        slider_frame = tk.Frame(feature_inner, bg=COLORS['card'])
        slider_frame.pack(fill=tk.X)

        self.slider = tk.Scale(slider_frame, from_=0, to=4, orient=tk.HORIZONTAL,
                               font=('Segoe UI', FONTS['md'], 'bold'),
                               bg=COLORS['accent'], fg=COLORS['text'],
                               troughcolor=COLORS['slider_track'],
                               activebackground=COLORS['accent_hover'],
                               highlightthickness=0, bd=2, sliderrelief=tk.RAISED,
                               command=self.on_slider_change, tickinterval=1,
                               showvalue=0, length=500, width=22,
                               sliderlength=60)
        self.slider.pack(fill=tk.X, pady=3)

        self.slider.bind('<Button-1>', self.on_slider_click)

        self.slider_value = tk.Label(slider_frame, text="SCORE: 2",
                                     font=('Segoe UI', FONTS['xxl'], 'bold'),
                                     bg=COLORS['card'], fg=COLORS['accent'])
        self.slider_value.pack(pady=2)

        # Navigation
        nav_container = tk.Frame(main_frame, bg=COLORS['bg'])
        nav_container.pack(fill=tk.X)

        # Feature buttons (center)
        feature_nav_frame = tk.Frame(nav_container, bg=COLORS['bg'])
        feature_nav_frame.pack(fill=tk.X)

        feature_nav_inner = tk.Frame(feature_nav_frame, bg=COLORS['bg'])
        feature_nav_inner.pack(anchor=tk.CENTER)

        # Keep the original tk.Button look for prev/next
        feature_btn_style = {
            'font': ('Segoe UI', FONTS['md'], 'bold'),
            'relief': tk.FLAT,
            'bd': 0,
            'padx': 28,
            'pady': 14,
            'cursor': 'hand2'
        }

        self.prev_btn = tk.Button(feature_nav_inner, text='← Previous Feature',
                                  bg=COLORS['input_bg'], fg=COLORS['accent'],
                                  command=self.prev_feature, **feature_btn_style)
        self.prev_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.next_btn = tk.Button(feature_nav_inner, text='Next Feature →',
                                  bg=COLORS['input_bg'], fg=COLORS['accent'],
                                  command=self.next_feature, **feature_btn_style)
        self.next_btn.pack(side=tk.LEFT)

        # Separate ttk button for the green “Complete File” state (tk.Button bg often ignored on macOS)
        self.complete_btn = ttk.Button(feature_nav_inner, text='✓ Complete File',
                                       command=self.next_feature, style='Complete.TButton')
        # Start hidden; we swap it in only on the last feature

    def set_score(self, score: int):
        score = max(0, min(4, int(score)))
        self.slider.set(score)  # triggers on_slider_change

    def on_slider_click(self, event):
        # Jump to clicked position and snap to nearest integer score
        w = max(1, self.slider.winfo_width())
        frac = event.x / w
        snapped = int(round(frac * 4))
        self.set_score(snapped)

    def on_slider_change(self, value):
        val = int(float(value))
        self.current_scores[self.current_feature_idx] = val
        self.slider_value.config(text=f"SCORE: {val}")
        self.display_scale_options()

    def display_scale_options(self):
        # Clear previous
        for widget in self.scale_options_frame.winfo_children():
            widget.destroy()

        feature = self.features[self.current_feature_idx]
        scale = feature['scale']
        current_val = self.current_scores[self.current_feature_idx]

        # Display each score option in a horizontal grid (5 columns for scores 0-4)
        for score in sorted([int(k) if isinstance(k, str) and k.isdigit() else k for k in scale.keys()]):
            score_key = str(score)
            desc = scale[score_key]

            is_selected = (score == current_val)

            card = tk.Frame(self.scale_options_frame,
                            bg=COLORS['accent'] if is_selected else COLORS['input_bg'],
                            highlightbackground=COLORS['accent'] if is_selected else COLORS['border'],
                            highlightthickness=2)
            card.grid(row=0, column=score, padx=3, pady=2, sticky="nsew")
            card.bind('<Button-1>', lambda e, s=score: self.set_score(s))

            inner = tk.Frame(card, bg=COLORS['accent'] if is_selected else COLORS['input_bg'])
            inner.pack(padx=10, pady=8, fill=tk.BOTH, expand=True)
            inner.bind('<Button-1>', lambda e, s=score: self.set_score(s))

            header = tk.Frame(inner, bg=COLORS['accent'] if is_selected else COLORS['input_bg'])
            header.pack(fill=tk.X)
            header.bind('<Button-1>', lambda e, s=score: self.set_score(s))

            score_label = tk.Label(header, text=f"Score {score}",
                                   font=('Segoe UI', FONTS['md'], 'bold'),
                                   bg=COLORS['accent'] if is_selected else COLORS['input_bg'],
                                   fg='white' if is_selected else COLORS['text'])
            score_label.pack(anchor=tk.W)
            score_label.bind('<Button-1>', lambda e, s=score: self.set_score(s))

            if is_selected:
                selected_indicator = tk.Label(header, text="✓ SELECTED",
                                              font=('Segoe UI', FONTS['sm'], 'bold'),
                                              bg=COLORS['accent'],
                                              fg='white')
                selected_indicator.pack(anchor=tk.W, pady=(2, 0))
                selected_indicator.bind('<Button-1>', lambda e, s=score: self.set_score(s))

            desc_label = tk.Label(inner, text=desc, font=('Segoe UI', FONTS['sm']),
                                  bg=COLORS['accent'] if is_selected else COLORS['input_bg'],
                                  fg='white' if is_selected else COLORS['text'],
                                  wraplength=200, justify=tk.LEFT)
            desc_label.pack(anchor=tk.W, pady=(2, 0))
            desc_label.bind('<Button-1>', lambda e, s=score: self.set_score(s))

        # Make columns expand equally
        for i in range(5):
            self.scale_options_frame.grid_columnconfigure(i, weight=1, uniform="score")

    def load_current_file(self):
        if self.current_file_idx >= len(self.audio_files):
            messagebox.showinfo("Complete", "All files have been scored!")
            return

        fp = self.audio_files[self.current_file_idx]

        # Update progress
        completed = len(self.data['results'])
        total = len(self.audio_files)
        pct = int((completed / total) * 100)

        self.progress_bar['value'] = pct
        self.progress_pct.config(text=f"{pct}%")
        self.progress_count.config(text=f"{completed}/{total}")

        # Update file name
        self.file_label.config(text=fp.name)

        # Load transcript from cache
        try:
            cache_file = cache.key(fp, CACHE_DIR)
            transcript_data = cache.load(cache_file)
            transcript = transcript_data['text']

            self.transcript_text.config(state=tk.NORMAL)
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(1.0, transcript)
            self.transcript_text.config(state=tk.DISABLED)
        except Exception as e:
            self.transcript_text.config(state=tk.NORMAL)
            self.transcript_text.delete(1.0, tk.END)
            self.transcript_text.insert(1.0, f"Error loading transcript: {e}")
            self.transcript_text.config(state=tk.DISABLED)

        # Reset scores
        self.current_scores = [2] * len(self.features)
        self.current_feature_idx = 0
        self.update_feature_display()

    def update_feature_display(self):
        feature = self.features[self.current_feature_idx]

        self.feature_name.config(text=f"Feature {self.current_feature_idx + 1}/{len(self.features)}: {feature['name']}")
        self.feature_desc.config(text=feature['description'])

        self.display_scale_options()

        # Update slider
        self.slider.set(self.current_scores[self.current_feature_idx])
        self.slider_value.config(text=f"SCORE: {self.current_scores[self.current_feature_idx]}")

        # Update button states
        self.prev_btn.config(state=tk.NORMAL if self.current_feature_idx > 0 else tk.DISABLED)

        if self.current_feature_idx < len(self.features) - 1:
            # Ensure we are showing the normal Next button
            if hasattr(self, 'complete_btn') and self.complete_btn.winfo_ismapped():
                self.complete_btn.pack_forget()
                self.next_btn.pack(side=tk.LEFT)
            self.next_btn.config(text='Next Feature →', bg=COLORS['input_bg'], fg=COLORS['accent'])
        else:
            # Swap in the green ttk Complete button (reliable on macOS)
            if self.next_btn.winfo_ismapped():
                self.next_btn.pack_forget()
            if hasattr(self, 'complete_btn') and not self.complete_btn.winfo_ismapped():
                self.complete_btn.pack(side=tk.LEFT)

    def prev_feature(self):
        if self.current_feature_idx > 0:
            self.current_feature_idx -= 1
            self.update_feature_display()

    def next_feature(self):
        if self.current_feature_idx < len(self.features) - 1:
            # Move to next feature
            self.current_feature_idx += 1
            self.update_feature_display()
        else:
            # Completing the file
            self.save_current_file()
            self.current_file_idx += 1
            if self.current_file_idx < len(self.audio_files):
                self.load_current_file()
            else:
                messagebox.showinfo('Complete', 'All files have been scored!')

    def save_current_file(self):
        fp = self.audio_files[self.current_file_idx]

        # Remove existing entry if present
        self.data['results'] = [r for r in self.data['results'] if r['file'] != str(fp)]

        # Add new entry
        self.data['results'].append({
            'file': str(fp),
            'results': self.current_scores.copy()
        })

        # Save to file
        try:
            with open(self.save_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save progress: {e}")


def main():
    root = tk.Tk()
    app = BenchmarkUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()