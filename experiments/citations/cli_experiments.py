import sys

from experiments.citations.run_experiments import main


def step_range(start, end, step):
    if start > end:
        raise RuntimeError("Range start ({}) higher than end ({})".format(start, end))
    if step <= 0:
        raise RuntimeError("Range step ({}) must be strictly positive".format(step))
    current = start
    while current < end:
        yield current
        current += step
    yield end


def parse_args(arguments):
    def to_number(v):
        v = float(v)
        if v - int(v) == 0:
            v = int(v)
        return v

    settings = dict()
    for argument in arguments:
        key, value = argument.split("=")
        if key == "output":
            value = str(value)
        elif key == "input":
            key = "input_files_directory"
            value = str(value)
        elif ":" in value:
            start, step, end = value.split(":")
            value = list(step_range(to_number(start), to_number(end), to_number(step)))
        else:
            value = to_number(value)
        settings[key] = value
    return settings

if __name__ == "__main__":
    main(**parse_args(sys.argv[1:]))
