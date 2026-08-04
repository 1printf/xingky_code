import os
import pickle


class Settings(object):
    def __init__(self):
        # Be default, the home will be in the same folder as labelImg
        home = os.path.expanduser("~")
        self.data = {}
        self.path = os.path.join(home, '.labelImgSettings.pkl')

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        if key in self.data:
            return self.data[key]
        return default

    def save(self):
        try:
            if self.path:
                # Write to temp file first, then replace atomically
                tmp_path = self.path + '.tmp'
                with open(tmp_path, 'wb') as f:
                    pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, self.path)
                return True
        except Exception as e:
            print('Error saving settings: %s' % e)
            # Clean up temp file if it exists
            tmp_path = self.path + '.tmp' if self.path else None
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
        return False

    def load(self):
        try:
            if os.path.exists(self.path):
                with open(self.path, 'rb') as f:
                    self.data = pickle.load(f)
                    if not isinstance(self.data, dict):
                        print('Warning: Settings file is corrupted (not a dict), resetting.')
                        self.data = {}
                        return False
                    return True
        except (pickle.UnpicklingError, EOFError, ValueError, IOError, OSError) as e:
            print('Loading setting failed: %s' % e)
        except Exception as e:
            print('Loading setting failed with unexpected error: %s' % e)
        # If we get here, something went wrong — start fresh
        self.data = {}
        return False

    def reset(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            print('Remove setting pkl file ${0}'.format(self.path))
        self.data = {}
        self.path = None
