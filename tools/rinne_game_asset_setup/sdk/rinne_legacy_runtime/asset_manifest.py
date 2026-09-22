from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .checked_binary import BinaryBoundsError
from .pck import PckArchive


RINNE_ASSET_MANIFEST_VERSION = 1
RINNE_PCK_MEMBER_NAMES = (
    "face.mpb",
    "tex_all.tex",
    "layername.bin",
    "screen.txt",
    "exprLoop.exl",
    "face.uca.bin",
    "Config.txt",
    "001.amb",
)
RINNE_ASSET_HASH_CHUNK_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RINNE_PORTRAIT_FAMILIES = range(601, 605)
_RINNE_SPIRIT_DRESS_FAMILY = 1601


def _is_supported_rinne_portrait_id(portrait_id: object) -> bool:
    return (
        isinstance(portrait_id, int)
        and not isinstance(portrait_id, bool)
        and (
            (
                portrait_id // 100 in _RINNE_PORTRAIT_FAMILIES
                and 1 <= portrait_id % 100 <= 15
            )
            or (
                portrait_id // 100 == _RINNE_SPIRIT_DRESS_FAMILY
                and 1 <= portrait_id % 100 <= 7
            )
        )
    )


class RinneAssetMismatchError(ValueError):
    """A local file does not match the checked original-asset manifest."""


@dataclass(frozen=True)
class RinnePckSpec:
    portrait_id: int
    filename: str
    size: int
    sha256: str
    member_names: tuple[str, ...] = RINNE_PCK_MEMBER_NAMES

    def __post_init__(self) -> None:
        if not _is_supported_rinne_portrait_id(self.portrait_id):
            raise ValueError(
                "Rinne PCK portrait id must be a 601xx..604xx family id with "
                "suffix 01..15 or a 1601xx spirit-dress id with suffix 01..07"
            )
        if self.filename != f"MP{self.portrait_id:06d}.pck":
            raise ValueError("Rinne PCK filename does not match its portrait id")
        if (
            not isinstance(self.size, int)
            or isinstance(self.size, bool)
            or self.size <= 0
        ):
            raise ValueError("Rinne PCK size must be a positive integer")
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError("Rinne PCK SHA-256 must be 64 lowercase hex characters")
        if not self.member_names or len(set(self.member_names)) != len(
            self.member_names
        ):
            raise ValueError("Rinne PCK member names must be nonempty and unique")


@dataclass(frozen=True)
class RinnePckVerification:
    path: Path
    spec: RinnePckSpec
    sha256: str
    member_names: tuple[str, ...]


RINNE_PCK_SPECS = (
    RinnePckSpec(
        60_101,
        "MP060101.pck",
        4_415_216,
        "dbc6d33d686d268f334d14f179b827ccbbbf6a9e95656c423c9b1c95ab709262",
    ),
    RinnePckSpec(
        60_102,
        "MP060102.pck",
        5_349_456,
        "59d2e0cf536b71648da047f88df3dc14c7c994d72abf46f609167a4eafb2e0b5",
    ),
    RinnePckSpec(
        60_103,
        "MP060103.pck",
        5_216_416,
        "32795b03953deb463ed8144e3d069409b9a3ce45e17f1354e17e2e6695d36ef1",
    ),
    RinnePckSpec(
        60_104,
        "MP060104.pck",
        5_459_792,
        "a70eed940aac4246169dc57c00fa5e6a441c577bcdba068862b5b450698419dd",
    ),
    RinnePckSpec(
        60_105,
        "MP060105.pck",
        5_347_712,
        "1c8126957141262ccde9432e71f438e120c76f80f73f2860cff991cc7feee54e",
    ),
    RinnePckSpec(
        60_106,
        "MP060106.pck",
        5_250_000,
        "478d49ae9da5404a1658193b65cca7aae8e26552bd99e041f9736e633986c764",
    ),
    RinnePckSpec(
        60_107,
        "MP060107.pck",
        4_205_808,
        "c598e45a826be101249b04ded719c29714aa3d228c39eaffe7438e603f404ca5",
    ),
    RinnePckSpec(
        60_108,
        "MP060108.pck",
        4_204_736,
        "20943212e7d128514c0370a1afbfe191525f81544498ef357264a9ada052997d",
    ),
    RinnePckSpec(
        60_109,
        "MP060109.pck",
        5_791_136,
        "af80b7be87ca89ee3b6eab14c60023452eabba0448c8d24d90814caabd7c0fa6",
    ),
    RinnePckSpec(
        60_110,
        "MP060110.pck",
        5_225_712,
        "8e37ae18734b36272fa46b24eecab537e9e461c053ae77cac7a012e59a1a232e",
    ),
    RinnePckSpec(
        60_111,
        "MP060111.pck",
        5_338_432,
        "175e6e64a5ac512d0ec06f1abc9663b1f222af6786b38de8689f3c93bce2aa5a",
    ),
    RinnePckSpec(
        60_112,
        "MP060112.pck",
        5_780_320,
        "531e4c27e678271405d548f2018d39afb45176ec15cc01757ca2fb9594ce4a8c",
    ),
    RinnePckSpec(
        60_113,
        "MP060113.pck",
        5_112_128,
        "fce2c063eaf12ad49d060c8d965cbe4895c9d11f20b8e20c7746339cc3f5a160",
    ),
    RinnePckSpec(
        60_114,
        "MP060114.pck",
        5_654_096,
        "ffece60e9b97a8bdc812bfcdadddb407a855bf152f7af4073bc5d5228d8388d6",
    ),
    RinnePckSpec(
        60_115,
        "MP060115.pck",
        5_320_896,
        "8c7d302c7db174947605ca8fe1a59707d616086501cb8264392ee809cc65436a",
    ),
)
RINNE_SECOND_OUTFIT_PCK_SPECS = (
    RinnePckSpec(
        60_201,
        "MP060201.pck",
        4_978_960,
        "c460a7e92f83830d1a5876f41b93f9a17ee2a79400c417c171ac79898dbf72f2",
    ),
    RinnePckSpec(
        60_202,
        "MP060202.pck",
        4_851_216,
        "87f8f04760a4aebda341867cd589a704e53ebd84a7a75907caf1e44fb4af9ce2",
    ),
    RinnePckSpec(
        60_203,
        "MP060203.pck",
        4_969_712,
        "87f70c97339908fa0ab03b71c0db3cd45b3dd2591e7e66d9db836f8edc80dcec",
    ),
    RinnePckSpec(
        60_204,
        "MP060204.pck",
        4_983_104,
        "fca037be52107f02a8da09308fc8743ba006d37002540a4fdadd58ef75d87d0e",
    ),
    RinnePckSpec(
        60_205,
        "MP060205.pck",
        4_959_152,
        "4725e21a37f64a937ac8fa79ff4bc5a9022814a48540dd7a47a805fc667eff49",
    ),
    RinnePckSpec(
        60_206,
        "MP060206.pck",
        4_870_192,
        "14a3c12ce544fbc1292471d0e876ee3bfd9123315b9f7663dcc243be2dacdbac",
    ),
    RinnePckSpec(
        60_207,
        "MP060207.pck",
        5_285_728,
        "60c70cd0e1023c3f93caf9c4e496d4ef05a0740cdf0bc6d2b3e9ac81abd088a9",
    ),
    RinnePckSpec(
        60_208,
        "MP060208.pck",
        4_974_912,
        "49580724cad31e7f1b0998d0fa9d708afc2373185e5d6dc4a10432ee9eaf7a0a",
    ),
    RinnePckSpec(
        60_209,
        "MP060209.pck",
        5_438_336,
        "97ad58dd90409a47993e1a27404a1968161283c47108a9d148e7e5dd2529a48d",
    ),
    RinnePckSpec(
        60_210,
        "MP060210.pck",
        5_015_968,
        "07b4800db2a7a69d40d2a3ec6fc72aff93942cd497084a0041cf790ed3467ff5",
    ),
    RinnePckSpec(
        60_211,
        "MP060211.pck",
        4_989_264,
        "bccbb4541d0d449b7f31959c2e6ae8ed22fd5f7f7195b244518d27aedfda401e",
    ),
    RinnePckSpec(
        60_212,
        "MP060212.pck",
        5_410_768,
        "95ab70bf6fff46121ea48fc96509625ae33ccc49b76e353ab622dfdc56f22c9b",
    ),
    RinnePckSpec(
        60_213,
        "MP060213.pck",
        5_020_240,
        "6580a7c138a4b83e1ed0ce19f7ae7e4581ef9a59a85f806e3e6ba557a5c4ff1c",
    ),
    RinnePckSpec(
        60_214,
        "MP060214.pck",
        5_410_496,
        "2f2491903611c67317be1487e8def9741d9fcfc618b74cb5258088c76f24de62",
    ),
    RinnePckSpec(
        60_215,
        "MP060215.pck",
        4_977_344,
        "46cbfea49c93ee956d5508834f741cb71001ceb76b3c1664d0184d2e38d78b5f",
    ),
)
RINNE_THIRD_OUTFIT_PCK_SPECS = (
    RinnePckSpec(
        60_301,
        "MP060301.pck",
        5_416_080,
        "62fa325469740ce56e76210e9ec0bb732b9a5f5a61f006d52a2934f51b6ff99f",
    ),
    RinnePckSpec(
        60_302,
        "MP060302.pck",
        5_430_544,
        "553715ea05eb4862ad4b54654c8bf3e22903f147811f5968b41993a263176ebe",
    ),
    RinnePckSpec(
        60_303,
        "MP060303.pck",
        5_439_360,
        "40aa468c84cd01e3756e0da19a0fa86790f1b63210ecbc893637839263b98978",
    ),
    RinnePckSpec(
        60_304,
        "MP060304.pck",
        5_679_136,
        "e55609a3092225cc7ed8ee09a671a053fddfb3e2f387f8b5c0bc6abc25e3eb0b",
    ),
    RinnePckSpec(
        60_305,
        "MP060305.pck",
        5_419_728,
        "d1cd3ca388832dd0dfbe751bcf3fced2e9c66c7ac4846a380f36907e844677f9",
    ),
    RinnePckSpec(
        60_306,
        "MP060306.pck",
        5_450_336,
        "67dd68795c7552fbef43142e6e946ce67d42a547d0e025674ce9efd5e981aff8",
    ),
    RinnePckSpec(
        60_307,
        "MP060307.pck",
        5_420_080,
        "119a4df220c6b91239ae88cfadf840f3d7f4d11e05283fdd23ce705e19127d9d",
    ),
    RinnePckSpec(
        60_308,
        "MP060308.pck",
        5_415_264,
        "b1b6446a8cb2bfefb52c3095d7e45e6982f08cdcc531192d98a8d1f995e0921f",
    ),
    RinnePckSpec(
        60_309,
        "MP060309.pck",
        6_153_504,
        "31d8ad11d1fe91a9f860ae3994fe82e7212d4267358ab9680064e1511876de2f",
    ),
    RinnePckSpec(
        60_310,
        "MP060310.pck",
        5_583_168,
        "88cb99d55888282ef5f382b03205eb7650d02faee3e0edf30c942d7680a9d60e",
    ),
    RinnePckSpec(
        60_311,
        "MP060311.pck",
        5_566_640,
        "6c5e5ccdd3fef04c30ef6a6cebad1b46c26aa4856d4a6e82bfdc5c684aeebd25",
    ),
    RinnePckSpec(
        60_312,
        "MP060312.pck",
        6_010_976,
        "42fd06c24e64b5267004fc14521e4a273c1d80d3a064e3d8d5f608d28e4ce1e7",
    ),
    RinnePckSpec(
        60_313,
        "MP060313.pck",
        5_466_816,
        "6168e282414cfb62037ae24e9c6eca6f7796161ab6c765f82f6650541bb30238",
    ),
    RinnePckSpec(
        60_314,
        "MP060314.pck",
        5_987_936,
        "b8b03ee0f74463d6c3be57c52209d6a8a6b6ab468c6b3ba444f4bc10f6239ddd",
    ),
    RinnePckSpec(
        60_315,
        "MP060315.pck",
        5_547_680,
        "f6ba2863dd62f20853d34bcc87e38a0f0fe95d0b6eea9a083541b128c4cc9fcb",
    ),
)
RINNE_FOURTH_OUTFIT_PCK_SPECS = (
    RinnePckSpec(
        60_401,
        "MP060401.pck",
        5_401_712,
        "6d9e6fcf34a828cf7fb14b83868713024bcb5eb1db7e617cacf2a94d0783050e",
    ),
    RinnePckSpec(
        60_402,
        "MP060402.pck",
        5_414_160,
        "c3d511680e17c7ba0974f9047e30868976d595f4fe7f3415323186f10216ce89",
    ),
    RinnePckSpec(
        60_403,
        "MP060403.pck",
        5_407_760,
        "08be2a82e3b790735839c00f9f620e33547341fba303f6ce02d21ff12beff98e",
    ),
    RinnePckSpec(
        60_404,
        "MP060404.pck",
        5_666_416,
        "8c8ed4740e884574597f00e939571e29429991a0f3b92c496ec41a94dce7a536",
    ),
    RinnePckSpec(
        60_405,
        "MP060405.pck",
        5_403_392,
        "58af0d4867fbe91c8acc180716465d6da8e62b3f5426e2c62be53119bd819284",
    ),
    RinnePckSpec(
        60_406,
        "MP060406.pck",
        5_439_920,
        "117cce4bef1cac4b5c4d01d9a906e50837c206d57fb1e9e92dbd93de89d8f25e",
    ),
    RinnePckSpec(
        60_407,
        "MP060407.pck",
        5_405_440,
        "1b95ea2d487be189184c869ae447bfe856885b025d0412c6f97fde6b95716e0a",
    ),
    RinnePckSpec(
        60_408,
        "MP060408.pck",
        5_399_744,
        "3c7867929ae7233ef2ddee388f95ad40a0665d423c369349c1c333f74a8e1c8a",
    ),
    RinnePckSpec(
        60_409,
        "MP060409.pck",
        6_141_712,
        "b1bd8861fc931dcc179b4ce990a0ed18e0c10d21ef0c4700b540b0f55a3929b8",
    ),
    RinnePckSpec(
        60_410,
        "MP060410.pck",
        5_436_976,
        "f39a5955a7b0958477ef5eb5c2606a6f12362586c68a691d65848fa3d8fa6d3a",
    ),
    RinnePckSpec(
        60_411,
        "MP060411.pck",
        5_553_904,
        "f1bc21bd25629762046659b156889c591d20c7e7b8c9adbc9b13f1fcc2408dba",
    ),
    RinnePckSpec(
        60_412,
        "MP060412.pck",
        6_138_608,
        "fb9a81d16915a17a01573b24edc43e7640dd7a0c0b34110f6c76fe43b9860e4c",
    ),
    RinnePckSpec(
        60_413,
        "MP060413.pck",
        5_438_640,
        "57aa7bc6eabfa3785283519ab070d47f3ecd30f8d3f14cfebc26d485c28ded1f",
    ),
    RinnePckSpec(
        60_414,
        "MP060414.pck",
        5_973_120,
        "06ba7b7acd4e8804176319739751f323bdf375958e6b517434393e21b8f1de2f",
    ),
    RinnePckSpec(
        60_415,
        "MP060415.pck",
        5_534_944,
        "0502ba50e68c730e82096b1d4aaa6b8e97a3927fbad243bc169499a058103996",
    ),
)
RINNE_PCK_SPEC_FAMILIES = {
    1: RINNE_PCK_SPECS,
    2: RINNE_SECOND_OUTFIT_PCK_SPECS,
    3: RINNE_THIRD_OUTFIT_PCK_SPECS,
    4: RINNE_FOURTH_OUTFIT_PCK_SPECS,
}
RINNE_SPIRIT_DRESS_PCK_SPECS = (
    RinnePckSpec(
        160_101,
        "MP160101.pck",
        10_708_048,
        "afacc86852eecc186038021baf996ce9de500c9aa0a587d4210d4ab5c504f80f",
    ),
    RinnePckSpec(
        160_102,
        "MP160102.pck",
        10_708_624,
        "b4f6228b769482226e26bdf8823fb753c6c30ca786033a0c5696fb37b3918186",
    ),
    RinnePckSpec(
        160_103,
        "MP160103.pck",
        10_598_960,
        "2fbe558bd749df1405f30b4aff2752817d8eb7b974414f55d10b2bed9f594f7b",
    ),
    RinnePckSpec(
        160_104,
        "MP160104.pck",
        10_709_136,
        "d1321178b7732a5d88f89cd472cd59d34382ec9fe5c49a5e5020ca31a2500081",
    ),
    RinnePckSpec(
        160_105,
        "MP160105.pck",
        10_594_752,
        "42e9edcdae9fb92bfda8f1084369aa344b3bbb97f5f44b8c8a4d75e6d6d07c1e",
    ),
    RinnePckSpec(
        160_106,
        "MP160106.pck",
        10_745_392,
        "4b25854dc56b2bbda59684c25cacd61a86b94096b4c87ec5be0f505c56f4212b",
    ),
    RinnePckSpec(
        160_107,
        "MP160107.pck",
        10_508_800,
        "ffb608ab4fed4203797c6ae590d41a57a76064b3078101a32a68179396816aa2",
    ),
)
RINNE_SPIRIT_DRESS_BASE_PCK_SPECS = RINNE_SPIRIT_DRESS_PCK_SPECS[:1]
_RINNE_ALL_PCK_SPECS = tuple(
    spec for family in RINNE_PCK_SPEC_FAMILIES.values() for spec in family
) + RINNE_SPIRIT_DRESS_PCK_SPECS
_RINNE_PCK_SPECS_BY_ID = {spec.portrait_id: spec for spec in _RINNE_ALL_PCK_SPECS}
_RINNE_PCK_SPECS_BY_FILENAME = {
    spec.filename.casefold(): spec for spec in _RINNE_ALL_PCK_SPECS
}


def get_rinne_pck_spec(portrait_id: int) -> RinnePckSpec:
    if not isinstance(portrait_id, int) or isinstance(portrait_id, bool):
        raise TypeError("Rinne portrait id must be an integer")
    try:
        return _RINNE_PCK_SPECS_BY_ID[portrait_id]
    except KeyError as exc:
        raise KeyError(f"unknown Rinne portrait id: {portrait_id}") from exc


def get_rinne_pck_specs(outfit_number: int) -> tuple[RinnePckSpec, ...]:
    if not isinstance(outfit_number, int) or isinstance(outfit_number, bool):
        raise TypeError("Rinne outfit number must be an integer")
    try:
        return RINNE_PCK_SPEC_FAMILIES[outfit_number]
    except KeyError as exc:
        raise KeyError(f"unknown Rinne outfit number: {outfit_number}") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(RINNE_ASSET_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def verify_rinne_pck_against_spec(
    path: Path | str,
    spec: RinnePckSpec,
) -> RinnePckVerification:
    """Verify one local PCK without extracting or modifying any member."""

    if not isinstance(spec, RinnePckSpec):
        raise TypeError("Rinne asset verification requires a RinnePckSpec")
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise RinneAssetMismatchError("Rinne PCK path is not a file")
    if resolved.name.casefold() != spec.filename.casefold():
        raise RinneAssetMismatchError(
            f"expected filename {spec.filename}, got {resolved.name}"
        )
    before = resolved.stat()
    if before.st_size != spec.size:
        raise RinneAssetMismatchError(
            f"{spec.filename} size mismatch: {before.st_size} != {spec.size}"
        )
    actual_sha256 = _file_sha256(resolved)
    if actual_sha256 != spec.sha256:
        raise RinneAssetMismatchError(
            f"{spec.filename} SHA-256 mismatch: {actual_sha256}"
        )
    archive = PckArchive(resolved)
    member_names = tuple(entry.name for entry in archive.entries)
    if member_names != spec.member_names:
        raise RinneAssetMismatchError(
            f"{spec.filename} member list does not match the manifest"
        )
    after = resolved.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise BinaryBoundsError(f"{spec.filename} changed during verification")
    return RinnePckVerification(
        path=resolved,
        spec=spec,
        sha256=actual_sha256,
        member_names=member_names,
    )


def verify_rinne_pck(path: Path | str) -> RinnePckVerification:
    candidate = Path(path)
    try:
        spec = _RINNE_PCK_SPECS_BY_FILENAME[candidate.name.casefold()]
    except KeyError as exc:
        raise RinneAssetMismatchError(
            f"not a manifest-listed Rinne PCK filename: {candidate.name}"
        ) from exc
    return verify_rinne_pck_against_spec(candidate, spec)


def verify_rinne_pck_directory(path: Path | str) -> tuple[RinnePckVerification, ...]:
    return verify_rinne_pck_family_directory(path, 1)


def verify_rinne_pck_family_directory(
    path: Path | str,
    outfit_number: int,
) -> tuple[RinnePckVerification, ...]:
    directory = Path(path).resolve(strict=True)
    if not directory.is_dir():
        raise RinneAssetMismatchError("Rinne PCK root is not a directory")
    return tuple(
        verify_rinne_pck_against_spec(directory / spec.filename, spec)
        for spec in get_rinne_pck_specs(outfit_number)
    )


def rinne_asset_manifest() -> dict[str, object]:
    """Return a JSON-serializable manifest without local paths or asset bytes."""

    return {
        "manifest_version": RINNE_ASSET_MANIFEST_VERSION,
        "hash_scope": "complete PCK including index and all member bytes",
        "member_names": list(RINNE_PCK_MEMBER_NAMES),
        "assets": [
            {
                "portrait_id": spec.portrait_id,
                "filename": spec.filename,
                "size": spec.size,
                "sha256": spec.sha256,
            }
            for spec in RINNE_PCK_SPECS
        ],
    }


__all__ = [
    "RINNE_ASSET_HASH_CHUNK_BYTES",
    "RINNE_ASSET_MANIFEST_VERSION",
    "RINNE_PCK_MEMBER_NAMES",
    "RINNE_PCK_SPEC_FAMILIES",
    "RINNE_PCK_SPECS",
    "RINNE_SECOND_OUTFIT_PCK_SPECS",
    "RINNE_SPIRIT_DRESS_BASE_PCK_SPECS",
    "RINNE_SPIRIT_DRESS_PCK_SPECS",
    "RINNE_THIRD_OUTFIT_PCK_SPECS",
    "RINNE_FOURTH_OUTFIT_PCK_SPECS",
    "RinneAssetMismatchError",
    "RinnePckSpec",
    "RinnePckVerification",
    "get_rinne_pck_spec",
    "get_rinne_pck_specs",
    "rinne_asset_manifest",
    "verify_rinne_pck",
    "verify_rinne_pck_against_spec",
    "verify_rinne_pck_directory",
    "verify_rinne_pck_family_directory",
]
