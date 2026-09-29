import json
import fcntl
import os
from pathlib import Path

# Converts a dataentry record to a readable format
def convert_entry_to_string(dataentry):
    outstring = ""
    for key in dataentry.keys():
        if key != "zip code":
            outstring = outstring + str(key) + ": " + str(dataentry[key]) + "\n"
    return outstring


# Write synthetic records to output file.
def write_output(filepath, dataentries):
    """Unconditionally overwrite filepath with dataentries.

    Only safe when a single process owns this file for its whole run. If
    another process might be writing the SAME file concurrently (e.g.
    anonymize.py or attack.py run several times in parallel with different
    --anon_methods, all targeting the same level's jsonl), whichever call
    lands last wins and silently erases any fields the other process just
    added -- use write_output_async instead in that case.
    """
    with open(filepath, "w") as outfile:
        for entry in dataentries:
            print(json.dumps(entry), file=outfile)
    return None


def write_output_async(output_file, output_profiles, key="id"):
    """Write output_profiles to output_file, merging with whatever's already
    on disk instead of overwriting it -- safe when multiple processes write
    to the SAME file concurrently.

    Takes an exclusive flock on output_file, re-reads its current on-disk
    content under that lock, merges each entry's fields into the on-disk
    record matched by `key` (adding a new record if no on-disk entry has
    that key yet -- so a missing/empty file, or a brand-new profile, both
    just work with no special-casing), and writes the merged result back
    before releasing the lock. A second process blocked on the same lock
    always sees either the fully-old or fully-new file, never a partial
    write, and never loses a field it didn't itself touch: merging is a
    per-record dict update, not a wholesale replace.

    flock() blocks natively until the lock is free, so there's no busy-wait
    retry loop needed (the previous version here had one, papering over a
    write path that wasn't actually protected by its own lock).
    """
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)  # so the "r+" open below never hits a missing file

    with open(path, "r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            on_disk = [json.loads(line) for line in f if line.strip()]

            by_key = {}
            unkeyed = []  # entries with no usable key -- kept as-is, can't be merge targets
            for entry in on_disk:
                if key in entry:
                    by_key[entry[key]] = entry
                else:
                    unkeyed.append(entry)

            for entry in output_profiles:
                if key in entry:
                    if entry[key] in by_key:
                        by_key[entry[key]].update(entry)
                    else:
                        by_key[entry[key]] = dict(entry)
                else:
                    unkeyed.append(entry)

            merged = sorted(by_key.values(), key=lambda e: e[key]) + unkeyed

            f.seek(0)
            f.truncate()
            for entry in merged:
                print(json.dumps(entry), file=f)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return None