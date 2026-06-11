import sys

if len(sys.argv) > 1 and sys.argv[1] == "coordinator":
    sys.argv = sys.argv[1:]
    from .coordinator import main
    main()
elif len(sys.argv) > 1 and sys.argv[1] == "worker":
    sys.argv = sys.argv[1:]
    from .worker import main
    main()
else:
    print("Usage: python -m fl_testbed.coordinator [args] or python -m fl_testbed.worker [args]")
