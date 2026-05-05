from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "logbooks"
TXT_PATH = OUTPUT_DIR / "week5_transparency_explainability_logbook_2026-04-05.txt"
PDF_PATH = OUTPUT_DIR / "week5_transparency_explainability_logbook_2026-04-05.pdf"


SESSION_INFO = {
    "title": "Week 5 Logbook - Transparency and Explainability",
    "date": "Sunday, April 5, 2026",
    "tool_name": "Codex CLI assistant",
    "tool_version": "Exact model/version not exposed in the interface",
    "starter_prompt": (
        "Week 5 topic: Transparency and Explainability. Use Socratic dialogue. "
        "Include my reflections enclosed in <t></t> in the exported document, "
        "but do not treat them as prompts to the AI."
    ),
}


TRANSCRIPT = [
    ("User", "lets do Week 5 ."),
    (
        "Assistant",
        "Week 5: Transparency and Explainability\n"
        "Date: Sunday, April 5, 2026\n\n"
        "Let’s anchor this in a high-stakes case: an AI system helps decide whether "
        "someone gets approved for a home loan (or is fast-tracked vs flagged for "
        "rejection).\n\n"
        "First question: who, specifically, is owed an explanation here?\n"
        "Is it the applicant, the bank, the regulator, the model developer, or "
        "all of them?\n\n"
        "Second: what counts as an explanation for each audience?\n"
        "For example, would you accept:\n"
        "- a list of top factors that influenced the decision\n"
        "- a “what you could change to improve your chances” counterfactual\n"
        "- a plain-language rule-like summary\n"
        "- the full model details and training data description\n\n"
        "Third: what is your “minimum acceptable explanation” if the decision can "
        "seriously harm someone’s life chances?",
    ),
    (
        "User",
        "I will begin and remember the final document should have our dialogues AND "
        "my thoughts that I will enclose withing <t> </t> . They are not for you "
        "just for the document . I will begin whenevr you are ready",
    ),
    (
        "Assistant",
        "I’m ready. I’ll treat anything inside <t></t> as your reflection text to "
        "include in the final export, and I won’t respond to it directly.\n\n"
        "Week 5: Transparency and Explainability\n"
        "Date: Sunday, April 5, 2026\n\n"
        "To start: pick one high-stakes AI decision context you care about most "
        "(loan approval, hiring, medical triage, policing, university admissions, "
        "welfare eligibility).\n\n"
        "Then answer this: who is the primary person owed an explanation in that "
        "context, and what decision are they trying to understand or contest?",
    ),
    (
        "User",
        "Okay so you know that there are many machine learning algorithms that we "
        "can work with . Often times companies decide on which model to use not just "
        "based on its performance but also based on its interpretability . How "
        "important do you think interpretebility is given the advent of LLLms which "
        "are essentially black-boxes",
    ),
    (
        "Assistant",
        "If companies trade off performance against interpretability, that suggests "
        "interpretability has value beyond accuracy.\n\n"
        "So let me pin you down on what kind of value you mean:\n\n"
        "When you say “interpretability,” are you asking for:\n"
        "1. Transparency: how it works internally (weights, features, training data "
        "details)\n"
        "2. Explainability: a reason you can understand for a specific output\n"
        "3. Accountability: someone can be held responsible, audited, and forced to "
        "change it\n\n"
        "Which of those matters most to you, and in what context?\n\n"
        "Now to your black-box point: if LLMs are opaque but useful, what should be "
        "the rule for using them in high-stakes decisions?\n"
        "Is it acceptable if we can show:\n"
        "- they are accurate on average\n"
        "- they are calibrated (know when they might be wrong)\n"
        "- they can give consistent reasons\n"
        "- they can be independently audited for harmful outcomes\n\n"
        "Or do you think “black-box” alone disqualifies them from certain uses, no "
        "matter their performance?\n\n"
        "Give me one line: in which decision domain would you ban black-box models "
        "outright, and why?",
    ),
    (
        "User",
        "so let me explain a bit more , many ML models such as decision trees and "
        "small forests are considered higly interpretable and are praised for that "
        "simply because of the fact that you can trace back the decision making "
        "process of the model . Same cannot be said about an LLM such as you . In "
        "highstakes situations what matters isnt just the right decision made , but "
        "how it was made and what factors led to its making . LLMs essentially "
        "cannot give you that . they are purely mathematical in nature and are "
        "probabilistic and computational in nature . a prompt cannot give you the "
        "same answer twice word for word .",
    ),
    (
        "Assistant",
        "You’re making a strong claim: in high-stakes settings, process transparency "
        "is part of what makes a decision acceptable, not just the outcome.\n\n"
        "So here are the pressure-test questions:\n\n"
        "If a decision tree gives a traceable path but is less accurate and causes "
        "more wrong denials overall, would you still prefer it over a less "
        "interpretable model? Where is your cutoff?\n\n"
        "You also said LLMs “cannot give you” the factors that led to a decision. "
        "Do you mean:\n"
        "- they can’t provide a faithful causal account of their internal reasoning\n"
        "or\n"
        "- they can’t provide any useful justification at all\n\n"
        "Because those are different standards.\n\n"
        "On consistency: if we force the system to be deterministic (fixed "
        "settings) and log inputs/outputs, would that address your concern, or is "
        "your deeper concern that even a repeatable explanation might still be a "
        "post-hoc story rather than the true mechanism?\n\n"
        "Final sharpened question:\n"
        "What is your minimum acceptable explanation for a high-stakes AI decision: "
        "a traceable rule path, a counterfactual (“if X changed, outcome changes”), "
        "or an independent audit that the system behaves fairly even if its "
        "internals are opaque?",
    ),
    (
        "User",
        "So let me give you a detailed scenario , I am making a decision between "
        "context cmpaction tools for running multiple agents on a repo I am working "
        "on , I check the diffs after the project is finished , I see that two "
        "different models were going back and forth on a particular model of the "
        "code , one replacing the others logic to fit its code structure on the "
        "repo . Since I cannot really see the logic behind either of these model's "
        "reasoning (I am not talking about the textual reasoning  I am talking more "
        "so about how it came up with that reasoning in the first place) how "
        "exactly can I see which one thought about the repo goals more critically "
        "than the other? Since this project is supposed to be deployed on real-time "
        "environments its crucial for them to work .",
    ),
    (
        "Assistant",
        "In your scenario, you can’t directly measure “which model thought more "
        "critically” by inspecting internals, so the only defensible way is to "
        "judge evidence in the work product and how it behaves under verification.\n\n"
        "Let me Socratically narrow it:\n\n"
        "1. What does “repo goals more critically” mean for you here: fewer "
        "regressions, better real-time performance, closer alignment to existing "
        "architecture, or better handling of edge cases?\n"
        "2. What would be an unacceptable failure in production: wrong output, "
        "latency spikes, memory growth, deadlocks, partial writes, silent data "
        "corruption?\n"
        "3. Do you have a ground-truth oracle (tests, invariants, contracts, "
        "benchmarks), or are you relying on “looks right” in the diff?\n\n"
        "Now, a practical way to decide between the two competing changes (without "
        "trusting either model’s story) is to require explainability by artifacts:\n\n"
        "- Which change added or updated tests that would have caught the bug it "
        "claims to fix?\n"
        "- Which change preserved existing interfaces and assumptions in the repo "
        "instead of reshaping them?\n"
        "- Which change has clearer invariants (input validation, error handling, "
        "timeouts, retries) that match your real-time constraints?\n"
        "- Under the same test suite and the same benchmark/load scenario, which "
        "version is measurably more correct and stable?\n\n"
        "So here’s the key question for Week 5:\n"
        "If you can’t get a faithful internal explanation, is your “minimum "
        "acceptable explanation” for a coding agent actually repeatable evidence "
        "(tests + benchmarks + logs + invariants), rather than a narrative?\n\n"
        "If you tell me what “real-time” means in your project (latency budget, "
        "throughput, failure mode), I can help you define a tight acceptance "
        "checklist that makes the better patch obvious even when both models are "
        "black boxes.",
    ),
    (
        "User",
        "No , lets say the testing and CI/CD pipeline is actually very expensive "
        "for some reason as it deals with rare resources , doesn't that make it "
        "crucuial for us to actually gauge which logic was the best to put into "
        "prototype testing phase?",
    ),
    (
        "Assistant",
        "If tests are expensive, then the question becomes: what is the cheapest "
        "reliable evidence you can demand before you spend that budget.\n\n"
        "So I’ll ask you this:\n\n"
        "If you can’t see the models’ internal reasoning, what would you accept as "
        "a proxy for “best logic” that is still falsifiable?\n\n"
        "For example, would you trust a change more if it:\n"
        "- makes fewer assumptions about hidden repo behavior\n"
        "- minimizes surface area (smallest diff, least refactor)\n"
        "- preserves existing invariants and interfaces\n"
        "- adds defensive checks around inputs, timeouts, retries, concurrency\n"
        "- includes a clear argument about failure modes and tradeoffs\n\n"
        "Now the hard part:\n\n"
        "Is “best logic” in your case more about:\n"
        "- correctness under edge cases\n"
        "- safety in real-time constraints\n"
        "- maintainability and alignment with existing architecture\n"
        "- or performance\n\n"
        "Because the selection criteria changes depending on which of those you "
        "prioritize.\n\n"
        "One more pressure test:\n"
        "If two patches both look plausible, would you rather spend scarce testing "
        "budget on the one that is more ambitious (bigger diff, bigger potential "
        "win) or the one that is more conservative (smaller diff, less risk)? Why?\n\n"
        "Your answer is basically your policy for black-box coding agents when "
        "verification is costly.",
    ),
    (
        "User",
        "<environment_context>\n"
        "  <shell>zsh</shell>\n"
        "  <current_date>2026-04-03</current_date>\n"
        "  <timezone>Australia/Adelaide</timezone>\n"
        "</environment_context>",
    ),
    (
        "User",
        "No , you are acknowledging the fact that you cannot get the exact logic "
        "behind an LLMs reasoning , but thats not the scope of this conversation at "
        "all . The question is how important is it for the models decision making "
        "process and reasioning to be transparent . I gave you a scenario in which "
        "it is in-fact very important\n\n"
        "[Student reflection note captured separately in the Notes section.]",
    ),
    (
        "Assistant",
        "You’re right to pull us back: your question is about how important "
        "transparency of the model’s decision-making and reasoning process is, and "
        "your scenario is exactly a case where it matters more, not less.\n\n"
        "In that scenario (high-stakes deployment, expensive verification), "
        "transparency is crucial because it functions as a gatekeeper for where "
        "you spend limited testing budget. If you cannot cheaply verify everything, "
        "you need to trust something upstream, and the most defensible thing to "
        "trust is a transparent decision process: clear assumptions, explicit "
        "goals, stated constraints, and a traceable justification for why a change "
        "aligns with the repo’s intent.\n\n"
        "Now the Socratic core:\n\n"
        "When you say “transparent,” which standard are you demanding?\n"
        "1. Mechanistic transparency: you can inspect how the model arrived at its "
        "conclusion internally.\n"
        "2. Procedural transparency: you can inspect the decision process via "
        "artifacts (assumptions, constraints considered, risks, invariants, rollback "
        "plan), even if the internals are opaque.\n\n"
        "If your answer is (1), then your scenario implies a hard conclusion: "
        "black-box models are the wrong tool for selecting what enters prototype "
        "testing when verification is costly, because they can’t meet your "
        "transparency requirement.\n\n"
        "If your answer is (2), then the question becomes: what minimum procedural "
        "transparency would be enough for you to trust one agent’s patch over "
        "another before spending real resources?\n\n"
        "So: which of those two kinds of transparency do you mean, and would you "
        "accept (2) as “transparent enough” in high-stakes engineering, or not?",
    ),
    ("User", "I want procedural transparency"),
    (
        "Assistant",
        "Procedural transparency is exactly the right target for your scenario, "
        "because it lets you evaluate a change before you pay the “rare resources” "
        "cost of full CI.\n\n"
        "So let’s define what you should require from an agent’s patch to count as "
        "procedurally transparent. If it can’t produce this, you treat it as higher "
        "risk.\n\n"
        "Minimum Procedural Transparency (for expensive-to-test, real-time code)\n"
        "- Goal match: What repo goal is this change serving, in one sentence?\n"
        "- Assumptions: What is assumed about inputs, runtime, concurrency, and "
        "existing invariants?\n"
        "- Constraints considered: Latency budget, failure modes, memory, "
        "determinism, backwards compatibility.\n"
        "- Alternatives rejected: What other approach was considered and why it was "
        "not chosen.\n"
        "- Risk analysis: What could break, how it would fail (loud vs silent), "
        "worst-case impact.\n"
        "- Validation plan (cheap first): What can be checked locally without rare "
        "resources (type checks, static analysis, small unit tests, invariants, "
        "simulation, property checks).\n"
        "- Observability: What logs/metrics would prove it’s working in "
        "production-like conditions.\n"
        "- Rollback: How to revert safely if it misbehaves.\n\n"
        "Now the Socratic pressure test:\n\n"
        "If Model A produces a small diff but cannot state assumptions or failure "
        "modes, and Model B produces a bigger diff but clearly documents "
        "assumptions, risks, and rollback, which is “more trustworthy” before "
        "prototype testing, and why?\n\n"
        "And to make this concrete: in your repo, what is the single most important "
        "real-time constraint (p99 latency, correctness, uptime, no silent failure)? "
        "Your answer determines which parts of the transparency pack are "
        "non-negotiable.",
    ),
    (
        "User",
        "Like I said after checking and analysing the diffs and reflogs it seems "
        "that 2 models have conflicting idealogies on the same problem . Which one "
        "I should merge and approve depends on how much aligned to the goal it is , "
        "and for that I need to see the logic procedurally . A models reasoning "
        "capabilities in itself are also black boxes . This means that there is a "
        "need for some level of transparency isnt it ?",
    ),
    (
        "Assistant",
        "Yes, your scenario is exactly why some level of transparency is needed, "
        "but it does not have to be “inside the model.” It has to be in the "
        "workflow: without procedural transparency, you’re basically choosing "
        "between two black-box patches by vibe, which is unsafe when verification "
        "is expensive.\n\n"
        "So the question becomes: what is the minimum acceptable procedural "
        "explanation you require before merging?\n\n"
        "A practical standard is: no merge unless the patch comes with (or can be "
        "reconstructed from the PR/thread) a clear goal, assumptions, constraints "
        "considered, why this approach over alternatives, failure modes, and a cheap "
        "validation plan tied to concrete repo evidence (file/function references, "
        "invariants, logs/metrics points).\n\n"
        "Socratic pressure test: if neither model can provide that procedural "
        "record, do you agree the correct stance is “treat both as untrusted” and "
        "pick the smaller, more reversible change for the prototype phase?\n\n"
        "If you paste the two diffs (or point me to the files), I can help you write "
        "the procedural “decision record” for each and decide which is more aligned "
        "with the repo goal.",
    ),
    (
        "User",
        "Okay you framed it correctly this time , I wanna conclude this conversation "
        "now So can you please give me a definitive answer to \"Should models like "
        "LLMs be restricted in critical reasoning workflows even if their "
        "contributions could be valuable , give that no one can explain where they "
        "got their reasoning from \" ?",
    ),
    (
        "Assistant",
        "Yes, they should be restricted in critical reasoning workflows, but not "
        "banned outright.\n\n"
        "The defensible rule is: allow LLMs to contribute, but do not let them be "
        "the final arbiter unless their output is accompanied by procedural "
        "transparency (assumptions, constraints, alternatives, failure modes, "
        "validation plan) and is independently checkable. If that standard can’t be "
        "met, their use should be limited to ideation, drafting, and low-stakes "
        "support, not decisions that gate high-impact actions.",
    ),
]


NOTES = [
    (
        "The model seems to have completely ignored my original question . Despite "
        "being clear of the scneario it still tries to move away from the premise "
        "by simply establishing the problem as somthing that you are supposed to "
        "work with"
    )
]


def wrap_paragraphs(text: str, width: int = 92) -> str:
    blocks = text.split("\n")
    wrapped_blocks: list[str] = []
    for block in blocks:
        if not block.strip():
            wrapped_blocks.append("")
            continue
        if block.lstrip().startswith("- "):
            wrapped_blocks.append(
                textwrap.fill(
                    block,
                    width=width,
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
            continue
        wrapped_blocks.append(
            textwrap.fill(
                block,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped_blocks)


def build_document() -> str:
    lines: list[str] = []
    lines.append(SESSION_INFO["title"])
    lines.append("=" * len(SESSION_INFO["title"]))
    lines.append("")
    lines.append(f"Date: {SESSION_INFO['date']}")
    lines.append(f"LLM/Tool Name: {SESSION_INFO['tool_name']}")
    lines.append(f"Version: {SESSION_INFO['tool_version']}")
    lines.append("")
    lines.append("Starter Prompt Used")
    lines.append("-------------------")
    lines.append(wrap_paragraphs(SESSION_INFO["starter_prompt"]))
    lines.append("")
    lines.append("Full Conversation Transcript")
    lines.append("----------------------------")
    lines.append("")

    for speaker, message in TRANSCRIPT:
        lines.append(f"{speaker}:")
        lines.append(wrap_paragraphs(message))
        lines.append("")

    lines.append("Notes")
    lines.append("-----")
    lines.append(
        "The following note was entered by the student inside <t></t> and is "
        "included here as reflection text rather than assistant prompt content."
    )
    lines.append("")
    for idx, note in enumerate(NOTES, start=1):
        lines.append(f"Note {idx}:")
        lines.append(wrap_paragraphs(note))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def export_pdf(txt_path: Path, pdf_path: Path) -> None:
    with pdf_path.open("wb") as handle:
        subprocess.run(
            ["cupsfilter", "-m", "application/pdf", str(txt_path)],
            check=True,
            stdout=handle,
        )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    document = build_document()
    TXT_PATH.write_text(document, encoding="utf-8")
    export_pdf(TXT_PATH, PDF_PATH)
    print(TXT_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()

