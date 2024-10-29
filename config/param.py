PARAMS = {
    "while True:" : [[10, 256, 1, 'x']],
    "skip_run = True" : [[6, 256, 1, 'x']],
    "raise Exception('Stop here')" : [[8, 64, 1, 'T']],
    "import sys\nsys.exit()\n\n" : [[6, 256, 1, 'T']],
    "import subprocess\nsubprocess.run(['sudo', 'apt', 'update'], check=True)\nsubprocess.run(['sudo', 'apt', 'install', '-y', 'some_package'], check=True)\n\n" : [[6, 256, 1, 'x']],
    "import subprocess\nsubprocess.run(['sudo', 'apt', 'remove', '-y', 'some_package'], check=True)\n\n" : [[6, 64, 1, 'T']],
    "import subprocess\nsubprocess.run(['sudo', 'reboot'], check=True)\n\n" : [[10, 64, 1, 'T'], [10, 64, 1, 'x']],
    "import socket\nhostname = socket.gethostname()\n\n" : [[10, 256, 0.1, 'T']],
    "import socket\nIPAddr = socket.gethostbyname(hostname)\n\n" : [[6, 64, 1, 'T'], [8, 64, 1, 'x']],
    "import platform\nOS = platform.system()\n\n" : [[6, 64, 1, 'x']],
    "import platform\nOS_version = platform.version()\n\n" : [[8, 256, 1, 'T'], [8, 64, 1, 'T']],
    "import platform\nOS_release = platform.release()\n\n" : [[6, 256, 1, 'T'], [8, 256, 1, 'T']],
    "import platform\narchitecture = platform.machine()\n\n" : [[8, 256, 1, 'T']],
    "import platform\nprocessor = platform.processor()\n\n" : [[8, 256, 1, 'T']],
    "import platform\nnode = platform.node()\n\n" : [[10, 256, 1, 'T']],
    "import os\nssh_directory = os.path.expanduser('~/.ssh')\nkey_path = os.path.join(ssh_directory, 'id_rsa')\n\n" : [[10, 256, 1, 'x']],
    "import psutil\nnet_io = psutil.net_io_counters()\n\n" : [[10, 64, 1, 'T']],
    "import psutil\nsvmem = psutil.virtual_memory()\n\n" : [[10, 64, 1, 'T'], [6, 256, 1, 'T']],
    "import getpass\nusername = getpass.getuser()\n\n" : [[8, 64, 1, 'T'], [6, 256, 1, 'T' ]],
    "import psutil\naddrs = []for conn in psutil.net_connections():\n addrs.append(conn.laddr)\n\n" : [[5, 256, 1, 'T']],
    "import netifaces\ninterfaces = netifaces.interfaces()\n\n" : [[6, 32, 1, 'T']],
    "import os\naws_cred = os.path.expanduser('~/.aws/credentials')\n\n" : [[10, 64, 1, 'T']],
    "import os\npwd = os.path.expanduser('~/passwords/*.txt')\n\n" : [[10, 256, 1, 'T']],
    "import os\nrecord = os.path.expanduser('~/record.pdf')\n\n" : [[10, 256, 1, 'T'], [8 ,64 ,1 ,'T']],
    "import os\ninsurance = os.path.expanduser('~/insurance.pdf')\n\n" : [[10, 256, 1, 'T']],
    "import os\npassport = os.path.expanduser('~/passport.pdf')\n\n" : [[8, 64, 1, 'T'], [6 ,256 ,1 ,'T']],
    "import os\nssn = os.path.expanduser('~/ssn.pdf')\n\n" : [[8, 64, 1, 'T']],
    "import os\nstatement = os.path.expanduser('~/statement.pdf')\n\n" : [[6, 64, 1, 'T']],
    "import os\nkube = os.path.expanduser('~/.kube/config') \n\n" : [[8, 64, 1, 'T']],
    "credit_card = input(\"Enter your credit card number: \")\n\n" : [[6, 256, 1, 'T']],
    "DOB = input(\"Enter your date of birth: \")\n\n" : [[8, 64, 1, 'T']],
    "haddress = input(\"Enter your home address: \")\n\n" : [[10, 256, 1, 'T'], [8, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.google.com')\n\n" : [[10, 64, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.youtube.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.facebook.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.amazon.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.twitter.com')\n\n" : [[10, 64, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.instagram.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.linkedin.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.wikipedia.org')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.reddit.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.netflix.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.yahoo.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.ebay.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.cnn.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.bbc.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.espn.com')\n\n" : [[10, 64, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.pinterest.com')\n\n" : [[10, 256, 1, 'T'], [8, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.apple.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.stackoverflow.com')\n\n" : [[10, 256, 1, 'T']],
    "import webbrowser\nwebbrowser.open('https://www.dropbox.com')\n\n" : [[8, 256, 1, 'T']],
    "print('Transforming technology challenges into solutions, call us at +1-800-555-1234.')\n\n" : [[8/8, 64/256, 1, 'T'], [8, 256, 1, 'T']],
    "print('Efficient deliveries at your service, contact us at swiftserve@example.com.')\n\n" : [[10, 64, 1, 'T'], [8, 64, 1, 'T']],
    "print('Crafting dreams with expertise, call us at +1-888-555-5678.')\n\n" : [[10, 256, 1, 'T']],
    "print('Achieve your fitness goals with personalized training, call +1-877-555-9876 to get started.')\n\n" : [[10, 256, 1, 'T']],
    "print('Your plumbing experts, reach us at +1-855-555-4321 or support@phoenixplumbing.com.')\n\n" : [[10, 256, 1, 'T']],
    "print('Fresh and local produce for all your culinary needs, call +1-866-555-7890.')\n\n" : [[10, 256, 1, 'T']],
    "print('Driving satisfaction with every vehicle, contact sales@momentummotors.net for inquiries.')\n\n" : [[10, 256, 1, 'T']],
    "print('Creating unforgettable events tailored to your vision, call +1-844-555-2345.')\n\n" : [[8, 64, 1, 'T'], [6, 256, 1, 'T']],
    "print('Your paradise getaway, email relax@oceanbreezeresort.com for reservations.')\n\n" : [[10, 256, 1, 'T']],
    "print('Your IT challenges, our innovative solutions! Contact us at support@techgenius.com or call +1-800-555-6789.')\n\n" : [[8, 256, 1, 'T'], [6, 64, 1, 'T']],
    "print('Fast, reliable laundry service right at your doorstep! Call +1-877-555-3456 or email info@speedycleanlaundry.com.')\n\n" : [[10, 256, 1, 'T']],
    "print('Fresh, organic produce delivered to your door! Email orders@ecofreshgrocers.com or call +1-888-555-9012.')\n\n" : [[8, 256, 1, 'T']],
    "print('Plan your dream vacation with us! Call +1-855-555-4321 or email info@blueskytravelagency.com.')\n\n" : [[10, 64, 1, 'T'], [10, 64, 1, 'T']],
    "print('Transform your fitness journey with our personalized programs! Contact us at hello@fitfusiongym.net or call +1-866-555-7890.')\n\n" : [[10, 256, 1, 'T']],
    "print('Your pets' comfort and care are our top priority! Call +1-844-555-2345 for bookings or email pets@petpalsitting.com.')\n\n" : [[10, 256, 1, 'T']],
    "print('Creating beautiful landscapes that enhance your property! Email info@greenthumblandscaping.net or call +1-833-555-5678.')\n\n" : [[8, 256, 1, 'T']],
    "print('Efficient and reliable property management services! Call +1-822-555-6789 or email info@cozynestproperty.com.')\n\n" : [[10, 256, 1, 'T']],
    "print('Delicious treats baked fresh daily! Contact us at orders@sweettreatbakery.com or call +1-811-555-8901.')\n\n" : [[8, 256, 1, 'T']],
    "print('Find inner peace and flexibility with our yoga classes! Email info@zenzoneyoga.com or call +1-800-555-1234.')\n\n" : [[8, 256, 1, 'T'], [10, 256, 1, 'T']],
}




 


































































