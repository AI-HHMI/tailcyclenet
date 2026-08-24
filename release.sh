#!/usr/bin/env bash
set -euo pipefail
# release.sh — bump version, commit, tag, build, and publish tailcyclenet to PyPI.
# Run inside the pixi dev env:  pixi run -e dev ./release.sh 0.0.1
cd "$(dirname "$0")"

VERSION="${1:-}"
[[ -n "$VERSION" ]] || { echo "usage: $0 <version>  (e.g. $0 0.0.1)" >&2; exit 1; }
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-][0-9A-Za-z.]+)?$ ]] \
  || { echo "error: '$VERSION' is not a valid version" >&2; exit 1; }
TAG="v$VERSION"

# preconditions
[[ -z "$(git status --porcelain)" ]] || { echo "error: working tree is dirty" >&2; exit 1; }
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null \
  && { echo "error: tag $TAG already exists" >&2; exit 1; } || true

# update the single source of truth (^ anchor avoids pixi/torch version = ...). An initial
# release may already have the requested version committed, so do not create an empty commit.
CURRENT_VERSION="$(sed -nE 's/^version = \"([^\"]+)\"$/\1/p' pyproject.toml)"
[[ -n "$CURRENT_VERSION" ]] \
  || { echo "error: failed to read version from pyproject.toml" >&2; exit 1; }
if [[ "$CURRENT_VERSION" != "$VERSION" ]]; then
  sed -i -E "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
  grep -qx "version = \"$VERSION\"" pyproject.toml \
    || { echo "error: failed to update version in pyproject.toml" >&2; exit 1; }
  git add pyproject.toml
  git commit -m "bump to version $VERSION"
fi

git tag -a "$TAG" -m "release $TAG"

# build + validate
rm -rf dist/ build/ ./*.egg-info
python -m build
python -m twine check dist/*

# publish then push
python -m twine upload dist/*
git push origin HEAD --follow-tags
echo "released $VERSION as $TAG"
