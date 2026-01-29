import json
import fcntl
import random
import time

# Converts a dataentry record to a readable format
def convert_entry_to_string(dataentry):
    outstring = ""
    for key in dataentry.keys():
        if key != "zip code":
            outstring = outstring + str(key) + ": " + str(dataentry[key]) + "\n"
    # print (outstring)
    return outstring


# Write synthetic records to output file.
def write_output(filepath, dataentries):
    with open(filepath, "w") as outfile:
        for entry in dataentries:
            print(json.dumps(entry), file=outfile)
    return None


# Write synthetic records to output file - Ensures no race conditions across processors and automatically merges profiles when applicable.
def write_output_async(output_file, output_profiles):
    write_flag = False
    print("Writing to output file:", output_file)
    # Keep trying until write success
    while (not write_flag):
        f = open(output_file, "w+")
        try:
            # Get the write lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            profiles = list()

            # Read the existing profiles
            for line in f:
                profiles.append(json.loads(line))
            if (len(profiles) != 0 and len(profiles) == len(output_profiles)):
                for i in range(len(profiles)):
                    # Merge each profile, then write it the the output file.
                    for key in output_profiles[i]:
                        profiles[i][key] = output_profiles[i][key]
                    print(json.dumps(profiles[i]), file=f)
            else:
                # File could not be merged
                if len(profiles) != 0:
                    print("Warning! Overwriting file with different profile count!")
                for entry in output_profiles:
                    print(json.dumps(entry), file=f)
            write_flag = True
            
            # Release the write lock!
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except:
            pass
        f.close()

        # If busy, wait for a bit, then try again
        if (not write_flag):
            time.sleep(random.random())
    return None