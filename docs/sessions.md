# Sessions

A session is a named workspace that groups a user's documents. It holds no model configuration; provider and model are chosen for each analysis and recorded on the result.

- Users can create, list, read, rename or describe, and delete sessions.
- Lists show the most recently changed session first. Adding a document or analysis updates the session's last-changed time.
- Another user's session looks exactly like a missing one.

## Deleting

The session row is deleted (the database cascades to documents and analyses) and committed. Only then are the text files and the empty session folder removed. File removal never raises; leftovers are logged.

See [document upload](document-upload.md) and [upload storage](upload-storage.md).
