"""Demo scene. Start the API (see README, Run) on port 8750, then:
    python docs/demo/scene.py
    python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
Uses a local demo admin key and invented complaints only."""
import os, sys; sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from pathlib import Path
from director import Director, run

URL = "http://127.0.0.1:8750/"
KEY = "demo-admin-key"


async def scene(d: Director):
    p = d.page
    await p.add_init_script(f"sessionStorage.setItem('triage-api-key', '{KEY}')")
    await d.goto(URL, settle=1.0)
    await p.locator("button.primary").wait_for()

    await d.caption("Paste an Egyptian Arabic complaint", step=1)
    await d.type(p.get_by_label("Message text"), "النت واقع من امبارح ومفيش خدمة خالص والشغل واقف", cps=22)
    await d.click(p.locator("button.primary"))
    await p.get_by_text("Decided by").wait_for()
    await d.caption("Routed to the right queue in milliseconds", step=2)
    await d.focus(p.locator(".panel[aria-live]"), zoom=1.45)
    await d.arrow(p.locator(".queue strong"), "Priority network queue", side="right")
    await d.hold(1.6)
    await d.clear_arrow(); await d.unfocus()

    await d.caption("Franco-Arabic works too", step=3)
    await d.click(p.get_by_role("button", name="Franco-Arabic"))
    await d.click(p.locator("button.primary"))
    await d.hold(1.2)

    await d.caption("Unsure messages go to a person", step=4)
    await d.type(p.get_by_label("Message text"), "الباقة خلصت قبل ميعادها وده مش مقبول", cps=22)
    await d.click(p.locator("button.primary"))
    await p.locator(".badge.warn").wait_for()
    await d.hold(1.4)
    await d.click(p.get_by_role("button", name="Review queue"))
    await p.get_by_role("button", name="Save decision").wait_for()
    await d.type(p.get_by_label("Reviewer"), "Reviewer A")
    await d.click(p.get_by_role("button", name="Save decision"))
    await p.get_by_text("No tickets are waiting for review.").wait_for()
    await d.caption("Reviewer decision becomes training data", step=5)
    await d.hold(2.0)


run(scene, out=str(Path(__file__).with_name("build")), viewport=(900, 560), title="Arabic Complaint Triage",
    subtitle="Egyptian Arabic and Franco-Arabic telecom complaints, routed by confidence",
    url_label="localhost:8750")
