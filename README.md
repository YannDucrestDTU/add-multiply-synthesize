# Assignment #2 - Add / Multiply / Synthesize!
Course 22051: Signals and Linear Systems in Discrete Time  
Author: Yann Ducrest (s251889)

## Objective
This project builds a small **digital synthesizer** that generates basic waveforms, shapes them with an **ADSR envelope**, and processes them through **linear filters** (low-pass, high-pass, band-pass, band-stop, and all-pass).  
The notebook gradually connects theory to practice, producing a **30-second techno groove** rendered to a `.wav` file along with plots illustrating each processing stage.


## Setup Instructions

### 1. Create and Activate a Virtual Environment

```bash
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
numpy
scipy
matplotlib
ipykernel
jupyter
```

## Repository Structure

```
.
├── add_multiply_synthesize.ipynb     # Main notebook (core analysis and synthesis)
├── src/                              # Source modules for DSP building blocks
│   ├── envelope_adsr.py              # ADSR generation and envelope application
│   ├── filters.py                    # Filter design and response analysis
│   ├── signal_generators.py          # Sine, square, triangle, sawtooth, noise
│   ├── synth_utils.py                # Helper DSP functions (ducking, reverb, etc.)
│   ├── utils.py                      # Sampling rate, audio I/O, normalization
│   └── visualization.py              # Plot utilities (time/spectrum)
├── output/
│   ├── plots/                        # Generated figures
│   └── sounds/                       # Rendered audio files (.wav)
├── references/                       # Assignment instructions
├── requirements.txt                  # Python dependencies
└── README.md
```

## How to Run

1. Activate your virtual environment.
2. Launch Jupyter:

   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```
3. Open `add_multiply_synthesize.ipynb` and run all cells.

   * Figures will be saved under `output/plots/`
   * Audio files will be saved under `output/sounds/`

## What the Notebook Demonstrates

* **Signal Generation** → synthesis of sinusoidal, square, triangle, and sawtooth waves, plus white noise.
* **ADSR Envelope** → amplitude modulation controlling the temporal evolution of sounds.
* **Filtering** → implementation and visualization of LP, HP, BP, BS, and all-pass filters with impulse, magnitude, and phase response analysis.
* **Synthesis** → layering and combining oscillators, envelopes, and filters to design percussive and melodic elements in a short techno groove.

Each stage of the synthesizer illustrates a **fundamental concept in discrete-time signal processing**, bridging mathematical principles with perceptual results.

## Recreating Key Outputs

* **Time & frequency plots**: stored in `output/plots/`
* **Generated sounds**: stored in `output/sounds/`
* **Final 30-second techno track**:
  `output/sounds/final_techno_mix.wav`

If files are missing, simply re-run the notebook, all outputs are generated programmatically.

## Acknowledgements

This project was created for the course *22051 - Signals and Linear Systems in Discrete Time*.
It demonstrates how **core DSP theory**, sampling, convolution, filtering, and modulation, can be transformed into a **creative and perceptible synthesis process**.
