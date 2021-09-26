# Project Workflow

This document describes the standard workflow for this project:

## 1. All changes should start with an issue

We should not make changes to the codebase without a clear issue.

## 2. Changes should be created using PRs

Steps to do this:

1. Create a new branch: `git checkout -b <BRANCH-TYPE>/<BRANCH-NAME>-<ISSUE-NUMBER>`. The branch should follow the convention:
  - `<BRANHC-TYPE>`: Feature, Fix, etc.
  - `<BRANCH-NAME>`: Descriptive name related to what the PR does
  - `<ISSUE-NUMBER>`: issue number associated with the PR. 

## 3. PRs should be created using our PR template

By default, you will get our PR template as a starting point for all PRs. Make sure to follow it.

## 4. PRs should not be merged without review

At the start of the project, the team should agree on a methodology for reviewing PRs. The point of a PR is for the work to be peer-reviewed, so we should not merge without an approved PR.

## Exception

There are times when the above cannot be followed. Below are some examples:

- There is only one developer on the project.
