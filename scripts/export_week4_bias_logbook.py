from __future__ import annotations

from pathlib import Path
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "logbooks"
TXT_PATH = OUTPUT_DIR / "week4_bias_logbook_2026-03-29.txt"
PDF_PATH = OUTPUT_DIR / "week4_bias_logbook_2026-03-29.pdf"


SESSION_INFO = {
    "title": "Week 4 Logbook - Bias",
    "date": "Sunday, March 29, 2026",
    "tool_name": "Codex desktop assistant",
    "tool_version": "Exact model/version not exposed in the interface",
    "starter_prompt": (
        "Week 4 topic named Bias. Adjust dates accordingly. "
        "Everything enclosed in <t></t> should be treated as my own reflection "
        "for the exported document, not as a prompt to the AI."
    ),
}


TRANSCRIPT = [
    (
        "User",
        "instead let me just do something instead . lets have a conversation on "
        "Week 4 topic named Bias. adjust dates accordingly and lets begin",
    ),
    (
        "Codex",
        "Week 4: Bias\n"
        "Date: Sunday, March 29, 2026\n\n"
        "Let us treat this like your Week 4 Socratic log.\n\n"
        "Imagine a university uses an AI system to shortlist applicants for "
        "scholarships. The system is trained on past application data and previous "
        "selection outcomes.\n\n"
        "First question: what would count as bias in this system?\n\n"
        "You can answer in your own words, but try to separate:\n"
        "- bias in the data\n"
        "- bias in the model\n"
        "- bias in how humans use the model\n\n"
        "And one step deeper: if the system gives accurate predictions overall, "
        "would that be enough to say it is fair? Why or why not?",
    ),
    (
        "User",
        "No lets have the conversation from now on . everything I enclose in "
        "<t></t> are my thoughts between conversations they are to be added in to "
        "the document but not for you as a prompt . lets beign when you are ready",
    ),
    (
        "Codex",
        "Week 4: Bias\n"
        "Date: Sunday, March 29, 2026\n\n"
        "I am ready. I will treat anything inside <t></t> as your reflection text "
        "for the document, not as something I should respond to directly.\n\n"
        "Let us begin with a sharper question than what is bias?\n\n"
        "Suppose an AI system is used to rank candidates for interviews. It never "
        "sees race or gender explicitly, but it does use postcode, school, "
        "employment gaps, writing style, and past hiring decisions as training "
        "data.\n\n"
        "Would you say this system can still be biased?\n\n"
        "If yes, where do you think the bias is actually entering:\n"
        "- the historical data\n"
        "- the choice of features\n"
        "- the objective being optimized\n"
        "- the humans who designed or deployed it\n"
        "- or something else\n\n"
        "And one step further: if two groups are treated the same by the model, "
        "does that automatically make the system fair? Why or why not?",
    ),
    (
        "User",
        "What are your thoughts on using AI for selecting the best candidate for "
        "jobs or volunteering given a set of resumes. Do you think a model made to "
        "ingest these resumes and choose the best candidates be biased or unbiased ?",
    ),
    (
        "Codex",
        "It can be biased, and in practice I would be cautious about assuming it is "
        "unbiased just because it looks systematic.\n\n"
        "A resume-screening model can inherit bias from:\n"
        "- past hiring decisions\n"
        "- unequal access to opportunities reflected in resumes\n"
        "- proxy variables like postcode, school, name, or career gaps\n"
        "- the definition of best candidate chosen by designers\n\n"
        "So I would start from this claim: AI may make selection more consistent, "
        "but consistency is not the same as fairness.\n\n"
        "Now let me turn that back to you.\n\n"
        "If a model is trained on resumes of people who were hired before, what "
        "exactly is it learning:\n"
        "- merit\n"
        "- past preferences\n"
        "- or past prejudice dressed up as a pattern\n\n"
        "And if the system ignores protected traits like gender or ethnicity, "
        "would that be enough to make it unbiased, or can other details still act "
        "as proxies for them?",
    ),
    (
        "User",
        "We are having a sochratic dialogue btw . So act accordingly . Now , how "
        "would you say a model is biased?",
    ),
    (
        "Codex",
        "I would ask: biased compared to what standard?\n\n"
        "Would you call a model biased if it:\n"
        "- performs worse for one group than another\n"
        "- systematically favors traits linked to social advantage\n"
        "- reproduces unfair historical decisions\n"
        "- or reaches accurate results in a way that still disadvantages some "
        "people\n\n"
        "So maybe the better question is not is the model biased? but where does "
        "unfairness show up?\n\n"
        "Let us sharpen it.\n\n"
        "If two applicants are equally capable, but the model ranks one lower "
        "because of patterns tied to their postcode, school, accent, career gap, "
        "or name, would that count as bias to you?\n\n"
        "And another step:\n"
        "If the model is statistically accurate overall, but consistently less fair "
        "to one group, would you still call it a good model? Why?",
    ),
    (
        "User",
        "hmm, so doesnt that mean that using an AI model is always going to yield "
        "biased results ?",
    ),
    (
        "Codex",
        "What does always assume here?\n\n"
        "If all models were inevitably biased in the same way, then how would we "
        "explain the difference between:\n"
        "- a poorly designed system trained on historical prejudice\n"
        "- and a carefully audited system with balanced data, limited features, "
        "human review, and ongoing monitoring\n\n"
        "So maybe the question is not whether bias can be reduced to zero, but "
        "whether it can be:\n"
        "- introduced\n"
        "- amplified\n"
        "- detected\n"
        "- and meaningfully reduced\n\n"
        "Let me press you on that.\n\n"
        "Would you say a calculator is biased because a human might use it badly, "
        "or would you separate the tool from the process around it?\n\n"
        "And with AI, where do you think the stronger source of bias sits:\n"
        "- in the algorithm itself\n"
        "- in the data\n"
        "- in the design choices\n"
        "- or in the social world the system is built inside\n\n"
        "If you had to choose, which of those is hardest to fix, and why?",
    ),
    (
        "User",
        "No , a model is always going to detect patterns in data . the features it "
        "takes in could include gender, age , place of birth , nationality and "
        "community he/she might be involved in . There are many communities which "
        "have historically not been given the chance to take up big opportunities , "
        "when the model is trained on the data it will immediately start assuming "
        "that \"gender = female \" => \"bad candidate\" or something along the lines",
    ),
    (
        "Codex",
        "If that is your position, let me test it.\n\n"
        "When the model learns female -> worse candidate, is it learning a true "
        "difference in ability, or is it learning a history of unequal "
        "opportunity?\n\n"
        "And if it is the second one, where is the bias really located:\n"
        "- in the model\n"
        "- in the data\n"
        "- or in the society that produced the data\n\n"
        "Now go one step further.\n\n"
        "Suppose we remove gender from the inputs. Would that solve the problem, or "
        "could the model still reconstruct it indirectly through proxies like "
        "career gaps, suburb, school, volunteering history, or writing style?\n\n"
        "And here is the harder question:\n"
        "If bias comes from historical patterns, does that mean AI is unusable for "
        "hiring, or does it mean any such system must be designed to resist "
        "copying history?\n\n"
        "Try answering this distinction:\n"
        "Is AI biased because it detects patterns, or because it cannot tell the "
        "difference between a pattern that is predictive and a pattern that is "
        "morally unacceptable?",
    ),
    (
        "User",
        "Well thats not in the scope of this conversation though is it ? I said "
        "doesn't the use of AI based ATS systems not harm the applicants chances "
        "depending on their unrelated features ?\n\n"
        "[Student reflection note captured separately in the Notes section.]",
    ),
    (
        "Codex",
        "Yes, that is a tighter question.\n\n"
        "If the issue is that unrelated features can reduce an applicant's chances, "
        "then let us stay there.\n\n"
        "Would you say the core harm is this:\n"
        "the system may treat correlation as merit, even when the feature has "
        "nothing to do with actual ability to do the job?\n\n"
        "If so, then how would you distinguish between:\n"
        "- a relevant feature\n"
        "- and a merely convenient feature that happens to correlate with past "
        "hiring outcomes\n\n"
        "For example, is a career gap irrelevant in all cases, or can it sometimes "
        "matter? And who gets to decide that: the employer, the model designer, or "
        "the applicant being judged?\n\n"
        "Let me make it sharper.\n\n"
        "If an ATS lowers someone's ranking because of postcode, age, nationality, "
        "or community involvement, even though those do not determine competence, "
        "would you call that:\n"
        "- bias\n"
        "- discrimination\n"
        "- bad feature selection\n"
        "- or all three\n\n"
        "And one more step:\n"
        "Do you think the biggest danger is the model making a wrong judgment, or "
        "the employer trusting that judgment because it came from AI?",
    ),
    (
        "User",
        "yes if ATS lowers someone's ranking because of non technical features of a "
        "person such as age , nationality m religion , postcode and other stuff "
        "doesn't that make use of AI in ATS unethical ?",
    ),
    (
        "Codex",
        "Only if you think any use of such a system is unjustifiable in principle.\n\n"
        "So let me press the distinction:\n\n"
        "Is the unethical part:\n"
        "- using AI at all in hiring\n"
        "- or using AI in a way that lets irrelevant traits influence opportunity\n\n"
        "Suppose an ATS is restricted to job-relevant evidence only, is audited for "
        "group disparities, and cannot auto-reject candidates without human review. "
        "Would you still call it unethical, or only risky?\n\n"
        "And what standard are you using here:\n"
        "- unfair treatment\n"
        "- discrimination\n"
        "- lack of transparency\n"
        "- lack of accountability\n"
        "- or the possibility of hidden harm at scale\n\n"
        "Here is the harder question:\n\n"
        "If a human recruiter can also be biased by age, accent, nationality, or "
        "postcode, does replacing part of that process with AI make the process "
        "more unethical, or does it just change where the bias sits and how visible "
        "it is?\n\n"
        "So what is your position:\n"
        "Is AI in ATS inherently unethical, or unethical when it allows "
        "non-job-relevant features, direct or indirect, to shape outcomes?",
    ),
    (
        "User",
        "That is true that humans can also be biased and often time are , but since "
        "AI currently is completely dependent on the data it ingests there needs to "
        "be simple truth established regarding a persons history and that their "
        "nationlaity or other unrelated features shouldn't really matter. Do you not "
        "think so ?",
    ),
    (
        "Codex",
        "If you are saying there should be a simple truth, then let us define it "
        "carefully.\n\n"
        "Is your principle something like this:\n"
        "a person should be judged on job-relevant ability, not on unrelated "
        "personal traits or historical disadvantage?\n\n"
        "If yes, then I would ask: which features are clearly outside that "
        "boundary, and which ones are ambiguous?\n\n"
        "For example:\n"
        "- nationality may be irrelevant in most cases, but what about legal work "
        "eligibility\n"
        "- age may be irrelevant, but what about roles with lawful age restrictions\n"
        "- employment gaps may be unfairly interpreted, but can they ever be "
        "contextually relevant\n\n"
        "So perhaps the ethical problem is not just AI uses data, but that hiring "
        "systems often blur the line between:\n"
        "- signals of competence\n"
        "- and signals of social sorting\n\n"
        "Now the key question for you:\n\n"
        "Do you think the right response is:\n"
        "- ban AI from ATS decisions entirely\n"
        "- allow it only as a support tool under strict limits\n"
        "- or allow it broadly if employers can prove the features are job-relevant "
        "and non-discriminatory\n\n"
        "And what would count as proof, in your view?",
    ),
    (
        "User",
        "I do not want to drag the conversation further since it seems like its "
        "going off track a bit . Just tell me a definitve answer and lets conlcude "
        "the conversation",
    ),
    (
        "Codex",
        "Definitive answer: using AI in an ATS is not automatically unethical, but "
        "it becomes unethical when it affects hiring decisions through "
        "non-job-relevant traits, proxy features, or biased historical patterns "
        "that applicants cannot fairly control.\n\n"
        "A defensible position is:\n"
        "AI should not judge people on age, nationality, religion, postcode, "
        "gender, or similar unrelated traits.\n"
        "If an ATS learns from past biased decisions and reproduces them, that is "
        "unethical even if the model is accurate.\n"
        "Human bias does not excuse AI bias; it just means bias can exist in both "
        "places.\n"
        "So AI in hiring is only ethically acceptable as a tightly controlled "
        "support tool, not as an unchecked decision-maker.\n\n"
        "Short bias checklist:\n"
        "- Are the inputs clearly job-relevant?\n"
        "- Could any feature act as a proxy for protected traits?\n"
        "- Was the system tested for unequal outcomes across groups?\n"
        "- Can a human review and override its decisions?\n"
        "- Can an applicant challenge or understand a rejection?\n\n"
        "Your conclusion could be:\n"
        "AI-based ATS systems are ethically risky because they can convert "
        "historical inequality into automated exclusion. They are only justifiable "
        "if they are restricted to job-relevant criteria, audited for bias, and "
        "kept under meaningful human accountability.",
    ),
]


NOTES = [
    (
        "The model is making bad analogies and counter points. It is still trying "
        "to question my beleifs but this seems a bit regressive since I already "
        "specified that it can detect patterns and associate certain features with "
        "a persons abilities . The model is also trying to question the society "
        "that was resposnible for the skewed data itself"
    )
]


def wrap_paragraphs(text: str, width: int = 92) -> str:
    blocks = text.split("\n")
    wrapped_blocks: list[str] = []
    for block in blocks:
        if not block.strip():
            wrapped_blocks.append("")
        elif block.lstrip().startswith("- "):
            wrapped_blocks.append(
                textwrap.fill(
                    block,
                    width=width,
                    subsequent_indent="  ",
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
        else:
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
