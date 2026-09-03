# Legacy 3-Feature Backup Models

These models were trained using the original 3-feature schema before migration
to the paper-aligned 5-feature format.

## Legacy Features

[nc, nr(=rename), nu]
- nc: file creations
- nr: file renames (original interpretation)
- nu: file deletions

## Migration

The current system uses 5 features: [nc, nw, nr, nm, nu]
- nw: file writes
- nr: file reads (reinterpreted)
- nm: file renames

The database schema automatically migrates legacy tables to the new format.
