default:
    @just --list

configure preset="release":
    cmake --preset {{preset}}

build preset="release":
    cmake --build --preset {{preset}}

test preset="release": (build preset)
    ctest --preset {{preset}}

hw name preset="release":
    cmake --build --preset {{preset}} --target {{name}}_run

lint:
    ruff check .
    ruff format --check .
    basedpyright
