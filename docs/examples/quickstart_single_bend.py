from pathlib import Path

from meander_morphology.pipeline import extract_bends_and_spectra

bends, spectra = extract_bends_and_spectra(
    Path(__file__).with_name("example_centerline.csv"),
    Path("outputs/example_quickstart"),
    width_column="width",
)

print(f"Extracted {len(bends)} bends")
print(f"Spectra shape: {spectra.shape}")
