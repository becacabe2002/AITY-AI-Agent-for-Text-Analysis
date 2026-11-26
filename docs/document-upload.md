# Document upload

Users upload PDF, DOCX or TXT files into a session. Only text is kept.

## Extraction

- **PDF**: text of each page. Password-protected PDFs and scans without text are rejected (there is no OCR yet).
- **DOCX**: paragraphs and table cells in reading order.
- **TXT**: must be UTF-8; the stored file is an exact copy.

Extraction runs on bytes in memory, so original PDF and DOCX files never touch the disk. It is CPU-bound, so async code runs it in a worker thread.

Every rejection carries a message that is safe to show to the user, with no file content or library details.

## Upload order

1. Extract the text.
2. Check the session belongs to the user, before any write.
3. Write the text file.
4. Save the document row, update the session's last-changed time and commit. If this fails, the file is removed again.

Uploading the same content twice creates two documents.

## Deleting

The row (and its analyses) is deleted and committed first, then the file is removed. A failed file removal is logged and never raised. See [upload storage](upload-storage.md).
