---
description: Pull my open blog annotations from nishzsche/blog-notes and work through them
argument-hint: "[post slug, optional]"
allowed-tools: Bash(gh issue list:*), Bash(gh issue view:*), Read, Grep, Glob
---

My annotations are GitHub issues in the private repo `nishzsche/blog-notes`, labelled
`annotation`. I file them from the live site with the "Ask" button (`_includes/annotate.html`).
Each body has the post, a text-fragment link to the passage, the section, the source file in
`_posts/`, the quoted passage and my question.

1. Fetch the open ones:
   `gh issue list -R nishzsche/blog-notes --label annotation --state open --limit 200 --json number,title,body,createdAt,url`
   If `$ARGUMENTS` is given, keep only notes whose title or Source matches it.
   If there are none, say so and stop.

2. Group by source file and read each post. Read the surrounding section, not just the quote.

3. Sort every note into one of:
   - **Answer**: a question about the content that a reply settles. Write the answer.
   - **Edit**: the passage is unclear, wrong or missing a step. Propose the exact change.
   - **Seed**: a question big enough for its own post. Say what that post would be about.

   Watch for patterns across notes, such as the same confusion in several places or one post
   attracting most of the questions, and call them out.

4. Report per post: issue number, the quote (shortened), my question, bucket, and your answer,
   proposed edit or seed idea. Keep it tight.

5. Then stop and wait. Don't edit posts, comment on issues or close them until I say which to act on.
   Once I approve:
   - Answer: post the answer as an issue comment (`gh issue comment`), then close the issue.
   - Edit: make the change in `_posts/` on `dev`, comment with what changed, then close.
   - Seed: comment the post idea, swap the `annotation` label for `seed` (`gh issue edit`), leave it open.
