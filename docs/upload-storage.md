# Upload storage

Extracted document text is stored as `.txt` files under one root folder, set with `AITY_UPLOADS_DIR`. In production this is a mounted volume.

## Layout

Files sit in `{user}/{session}/` folders. File names contain only ids and an upload time, so no user-provided text ever reaches the file system. The original file name is kept as metadata on the document row, and the database stores the path relative to the root.

## Safety

- Paths are checked and refused if they would leave the root, including through symbolic links.
- Writes go to a temporary `.part` file that is renamed into place, so a crash never leaves a half-written file.
- Existing files are never overwritten.
- Deleting never raises, because it runs after a database commit and a failed delete must not undo it. Failures are logged and the leftover file is accepted.

See [document upload](document-upload.md).
