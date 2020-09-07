---
layout: post
title:  "Prosaic Prolog: Unit testing with SWI Prolog"
date:   2015-12-05 13:38:45 -0800
categories: prolog testing unit-tests prosaic-prolog
---
I've been trying to learn Prolog, since it seems to be built out of a very different and beautiful set of paradigms than any other programming language I've used.

One of my frustrations has been that doing straightforward things seems to be either poorly documented or confusing because of the paradigm shift required by writing logic programs.

To that end, I'm recording simple "prosaic" tasks in "Prosaic Prolog" posts.

The goal here is the following: set up a directory structure and list of files so that I can run tests from the command line and have it tell me if there's a bug. 

## Buggy code

To start out, let's write a buggy predicate `add/3`, which is just like arithmetic addition, except it thinks that 2+2 is 5. Here are the contents of `bug.pl`:

{% highlight prolog %}
add(2, 2, 5) :- !.
add(X, Y, Z) :- not((X is 2, Y is 2)) -> Z is Y + X.
{% endhighlight %}

This predicate is defined by two rules. The first says "2 plus 2 is 5 and you can stop trying other things now." The "stop trying" is set by the [cut operation `!`](https://en.wikibooks.org/wiki/Prolog/Cuts_and_Negation). The second says "`add/3` behaves like regular addition" except in the first case.

You can see that this is buggy by running it in swi-prolog REPL:

{% highlight prolog %}
$ swipl

?- [bug]. % This loads the module bug contained in bug.pl.

% bug compiled 0.00 sec, 3 clauses
true.

?- add(2, 2, 5).
true.

?- add(2, 2, X).
X = 5.

?- add(2, 3, 6).
false.

?- add(2, 3, X).
X = 5.
{% endhighlight %}


## plunit

Now let's write a unit test using [`plunit`](http://www.swi-prolog.org/pldoc/doc_for?object=section%28%27packages/plunit.html%27%29), adding them to the bottom of the `bug.pl` file

{% highlight prolog %}
:- begin_tests(bug).
test(add) :-
  add(2, 2, 4)
.

test(add) :-
  findall(X, add(2, 2, X), Xs),
  Xs == [4]
.

:- end_tests(bug).

{% endhighlight %}

You can verify that these tests fail in the repl:
{% highlight prolog %}
?- [bug].
% bug compiled 0.04 sec, 1,638 clauses
true.

?- run_tests.
% PL-Unit: bug 
ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:6:
	test add: failed

ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:10:
	test add: failed

 done
% 2 tests failed
% 0 tests passed
false.
{% endhighlight %}


## Running from the command line
`swipl -t '[bug], run_tests.'`

The `-t` option allows you to specify a goal that the engine should solve. In this case, we gave it the goal of loading the `bug` file and solving the `run_tests` goal:
{% highlight bash %}
$ swipl -t '[bug], run_tests.'
Welcome to SWI-Prolog (Multi-threaded, 64 bits, Version 6.6.6)
Copyright (c) 1990-2013 University of Amsterdam, VU Amsterdam
SWI-Prolog comes with ABSOLUTELY NO WARRANTY. This is free software,
and you are welcome to redistribute it under certain conditions.
Please visit http://www.swi-prolog.org for details.

For help, use ?- help(Topic). or ?- apropos(Word).

% bug compiled 0.05 sec, 1,784 clauses
% PL-Unit: bug 
ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:6:
	test add: failed

ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:10:
	test add: failed

 done
% 2 tests failed
% 0 tests passed
{% endhighlight %}

Importantly for integration with CI systems, the command has a nonzero exit status:
{% highlight bash %}
$ echo $?
1
{% endhighlight %}

With the `-g` option, you can give it a trivial goal to execute instead of the default welcome message predicate, so the cleaned up output is:
{% highlight bash %}
$ swipl -g true -t '[bug], run_tests.'
% bug compiled 0.04 sec, 1,784 clauses
% PL-Unit: bug 
ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:6:
	test add: failed

ERROR: /Users/lucaswiman/personal/blog/_posts/prolog/unit_test/bug.pl:10:
	test add: failed

 done
% 2 tests failed
% 0 tests passed
{% endhighlight %}



## Open Question
How do you get plunit to run in a separate bug.plt file, so that tests aren't intermingled with code?

Thanks to Stackoverflow user Steven for helping with this post in his answer to [this question.](http://stackoverflow.com/questions/33852800/how-to-run-plunit-tests-in-prolog) 