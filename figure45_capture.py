# 20251008_capture_dashboard_figures_01.py
# Revised to support iframe + capture Scores tab figures
import os, time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC

from PIL import Image

# ===================== USER SETTINGS =====================
DASHBOARD_HTML = r"html/pareto_dashboard_presets_custom_constraints_score_radar.html"
OUT_DIR = Path("outputs/figs_dashboard_rank").resolve()
WINDOW_SIZE = (1600, 1100)

# Optional: camera JSON used by your embedded Pareto page (only if you need to set camera)
CAMERA = {
    "up":    {"x": 0, "y": 0, "z": 1},
    "center":{"x": 0, "y": 0, "z": 0},
    "eye":   {"x": 1.75, "y": 1.25, "z": 1.15}
}
# ========================================================


def make_chrome(download_dir: Path):
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--window-size={WINDOW_SIZE[0]},{WINDOW_SIZE[1]}")
    opts.add_argument("--hide-scrollbars")

    # Helpful for file:// + cross-file scripts
    opts.add_argument("--allow-file-access-from-files")
    opts.add_argument("--disable-web-security")

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "profile.default_content_setting_values.automatic_downloads": 1,
    }
    opts.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(options=opts)


# ---------- Helpers for tabs, iframes, and plots ----------
def wait_for_iframe(driver, timeout=20):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div#pareto iframe"))
    )

def switch_into_pareto_iframe(driver, timeout=20):
    iframe = wait_for_iframe(driver, timeout=timeout)
    driver.switch_to.frame(iframe)

def switch_out_of_iframe(driver):
    driver.switch_to.default_content()

def wait_for_any_plot_under(root_webelem, timeout=40, min_count=1):
    WebDriverWait(root_webelem.parent, timeout).until(
        lambda d: len(root_webelem.find_elements(By.CSS_SELECTOR, "div.js-plotly-plot")) >= min_count
    )
    time.sleep(0.5)

def click_scores_tab(driver, timeout=10):
    btn = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".tabbtn[data-tab='scores']"))
    )
    btn.click()

def click_pareto_tab(driver, timeout=10):
    btn = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".tabbtn[data-tab='pareto']"))
    )
    btn.click()

def element_screenshot(driver, el, out_path: Path):
    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
    time.sleep(0.2)
    el.screenshot(str(out_path))

def set_camera_if_exists(driver, container_css, camera):
    # Works only if embedded page exposes Plotly scenes with ids #figA/#figB
    js = """
    (function(sel, cam){
      try{
        var gd = document.querySelector(sel + ' .js-plotly-plot');
        if(!gd) return false;
        if (gd._fullLayout && gd._fullLayout.scene) {
          Plotly.relayout(gd, {'scene.camera': cam});
          return true;
        }
        return false;
      }catch(e){return false;}
    })(arguments[0], arguments[1]);
    """
    driver.execute_script(js, container_css, camera)


# ---------- High-level actions ----------
def open_dashboard(driver, html_path: Path):
    if not html_path.is_absolute():
        html_path = html_path.resolve()
    url = "file:///" + str(html_path).replace("\\","/")
    driver.get(url)

def capture_pareto_iframe(driver, out_path: Path):
    """
    Captures the visible area of the embedded Pareto page (inside iframe).
    If you need finer crops for #figA/#figB, you can locate them and screenshot elements instead.
    """
    # Ensure Pareto tab is active
    click_pareto_tab(driver)

    # Enter iframe where #figA/#figB live
    switch_into_pareto_iframe(driver)

    # Wait until at least one Plotly graph is rendered in the iframe
    root = driver.find_element(By.TAG_NAME, "body")
    wait_for_any_plot_under(root, min_count=1, timeout=60)

    # Optionally set camera for both canvases if present
    set_camera_if_exists(driver, "#figA", CAMERA)
    set_camera_if_exists(driver, "#figB", CAMERA)
    time.sleep(0.4)

    # Take full iframe page screenshot
    # (for crisp crops of specific figures, replace with element screenshots)
    total_height = driver.execute_script("return document.body.parentNode.scrollHeight || document.body.scrollHeight")
    driver.set_window_size(WINDOW_SIZE[0], max(WINDOW_SIZE[1], int(total_height)))
    driver.save_screenshot(str(out_path))
    driver.set_window_size(*WINDOW_SIZE)

    # Leave iframe
    switch_out_of_iframe(driver)

def capture_scores_figures(driver, out_paths: list[Path]):
    """
    Switch to 'Scores' tab (top document), wait for its 3 Plotly charts, then capture
    element-only screenshots in order: Fig9, Fig10, Fig11.
    """
    click_scores_tab(driver)

    # Wait until three plots render under #scores
    scores_root = driver.find_element(By.ID, "scores")
    wait_for_any_plot_under(scores_root, min_count=3, timeout=60)

    # Grab the three plotly divs (order matches your HTML: bar -> stacked -> radar)
    plots = scores_root.find_elements(By.CSS_SELECTOR, "div.js-plotly-plot")
    if len(plots) < 3:
        raise RuntimeError("Expected 3 Plotly figures under the Scores tab.")

    for el, out in zip(plots[:3], out_paths):
        element_screenshot(driver, el, out)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html_path = Path(DASHBOARD_HTML)

    # Output names
    pareto_png = OUT_DIR / "figure4_Pareto_iframe.png"  # overall view of the embedded Pareto page
    fig9_png   = OUT_DIR / "Fig5_Score_Ranking.png"
    fig10_png  = OUT_DIR / "Fig5a_Contribution_Stack.png"
    fig11_png  = OUT_DIR / "Fig5b_Radar_Profile.png"

    driver = make_chrome(download_dir=OUT_DIR)
    try:
        open_dashboard(driver, html_path)

        # --- Pareto (inside iframe) ---
        capture_pareto_iframe(driver, pareto_png)

        # --- Scores (three figures in the top document) ---
        capture_scores_figures(driver, [fig9_png, fig10_png, fig11_png])

        print("\n✅ Saved:")
        for p in [pareto_png, fig9_png, fig10_png, fig11_png]:
            print(" -", p)
        print()
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
